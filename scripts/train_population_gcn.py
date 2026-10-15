"""
Population Graph Convolutional Network (Population GCN) cho phân loại ASD
Dựa trên bài báo:
    Parisot et al. "Disease Prediction using Graph Convolutional Networks:
    Application to Autism Spectrum Disorder and Alzheimer's Disease"
    Medical Image Analysis (2018) / arXiv:1806.01738

Các chức năng chính:
1. Trích xuất đặc trưng FC (19,900 chiều) từ ma trận tương quan ROI CC200.
2. Tích hợp dữ liệu Phenotypic (SITE_ID, SEX, AGE_AT_SCAN) để xây dựng ma trận kề dân số W.
3. Sparsify đồ thị theo k-NN để tránh over-smoothing.
4. Triển khai cả GCN (Kipf-Welling / Spectral) và GAT (Graph Attention).
5. Thực hiện 10-Fold và 5-Fold Stratified Cross-Validation với Fold-wise Feature Selection (chống Data Leakage).
"""

import os
import sys
import glob
import argparse

# Đảm bảo in UTF-8 không lỗi trên Windows PowerShell (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

# Kiểm tra thư viện PyG; nếu chưa có, dùng pure PyTorch fallback cho GCN
try:
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("ℹ️ torch_geometric chưa được cài đặt. Sẽ sử dụng native PyTorch GCN implementation.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Native PyTorch GCN Layer (Fallback nếu không có PyG)
# ─────────────────────────────────────────────────────────────────────────────
class NativeGCNConv(nn.Module):
    """
    Kipf & Welling GCN Layer chuẩn: H' = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H W
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj_norm):
        # adj_norm: (N, N) ma trận kề đã chuẩn hóa đối xứng
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj_norm, support)
        if self.bias is not None:
            output = output + self.bias
        return output


# ─────────────────────────────────────────────────────────────────────────────
# 2. Định nghĩa Mô hình
# ─────────────────────────────────────────────────────────────────────────────
class PopulationGCN(nn.Module):
    """
    Population GCN 2 lớp ẩn theo cấu hình của Parisot et al. (2018).
    """
    def __init__(self, in_features, hidden_dim1=64, hidden_dim2=32, num_classes=2, dropout=0.3):
        super().__init__()
        self.use_pyg = HAS_PYG
        self.dropout = dropout

        if HAS_PYG:
            self.conv1 = GCNConv(in_features, hidden_dim1)
            self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        else:
            self.conv1 = NativeGCNConv(in_features, hidden_dim1)
            self.conv2 = NativeGCNConv(hidden_dim1, hidden_dim2)

        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.classifier = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x, graph_repr):
        # graph_repr là edge_index (nếu PyG) hoặc adj_norm (nếu Native PyTorch)
        if self.use_pyg:
            h = self.conv1(x, graph_repr)
        else:
            h = self.conv1(x, graph_repr)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        if self.use_pyg:
            h = self.conv2(h, graph_repr)
        else:
            h = self.conv2(h, graph_repr)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.classifier(h)
        return out


class PopulationGAT(nn.Module):
    """
    Population Graph Attention Network (GAT) sử dụng multi-head attention.
    """
    def __init__(self, in_features, hidden_dim1=32, num_heads=4, hidden_dim2=32, num_classes=2, dropout=0.3):
        super().__init__()
        assert HAS_PYG, "GAT yêu cầu thư viện torch_geometric!"
        self.dropout = dropout
        self.gat1 = GATConv(in_features, hidden_dim1, heads=num_heads, concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim1 * num_heads)
        self.gat2 = GATConv(hidden_dim1 * num_heads, hidden_dim2, heads=1, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.classifier = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.gat2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.classifier(h)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Thuật toán Xây dựng Population Graph (Parisot et al., 2018)
# ─────────────────────────────────────────────────────────────────────────────
def compute_population_adjacency(features, pheno_df, k_nn=10, age_threshold=2.0):
    """
    Tính ma trận kề dân số W(i, j) theo công thức Parisot et al.:
    W(i, j) = CosineSim(x_i, x_j) * [ gamma_site(i, j) + gamma_sex(i, j) + gamma_age(i, j) ]
    sau đó sparsify bằng k-NN.
    """
    n_nodes = len(features)
    print(f"📊 Đang xây dựng Population Graph cho {n_nodes} đối tượng...")

    # 1. Cosine similarity của imaging features
    norm_feat = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sim_matrix = np.dot(norm_feat, norm_feat.T)
    # Scale về [0, 1]
    sim_matrix = 0.5 * (sim_matrix + 1.0)

    # 2. Phenotypic similarity measures
    sites = pheno_df['SITE_ID'].values
    sexes = pheno_df['SEX'].values
    ages  = pheno_df['AGE_AT_SCAN'].values

    # gamma_site: 1 nếu cùng site, 0 nếu khác
    gamma_site = (sites[:, None] == sites[None, :]).astype(float)

    # gamma_sex: 1 nếu cùng giới tính, 0 nếu khác
    gamma_sex  = (sexes[:, None] == sexes[None, :]).astype(float)

    # gamma_age: 1 nếu chênh lệch tuổi <= age_threshold
    age_diff   = np.abs(ages[:, None] - ages[None, :])
    gamma_age  = (age_diff <= age_threshold).astype(float)

    # Tổng trọng số phenotypic
    pheno_weight = gamma_site + gamma_sex + gamma_age

    # Ma trận tương đồng tổng hợp
    W = sim_matrix * pheno_weight
    np.fill_diagonal(W, 0.0)

    # 3. k-NN Sparsification
    adj_sparse = np.zeros_like(W)
    for i in range(n_nodes):
        # Lấy top-k láng giềng có trọng số cao nhất
        top_k_idx = np.argsort(W[i])[-k_nn:]
        adj_sparse[i, top_k_idx] = W[i, top_k_idx]

    # Đối xứng hóa: A = max(A, A^T)
    adj_symmetric = np.maximum(adj_sparse, adj_sparse.T)

    # Thêm self-loops
    np.fill_diagonal(adj_symmetric, 1.0)

    print(f"✅ Đồ thị hoàn thành: {n_nodes} nút, mật độ cạnh: "
          f"{np.count_nonzero(adj_symmetric) / (n_nodes * n_nodes):.2%}")

    return adj_symmetric


def normalize_adjacency_matrix(adj):
    """
    Chuẩn hóa đối xứng ma trận kề: D^{-1/2} A D^{-1/2}
    """
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Huấn luyện và Đánh giá (Cross-Validation)
# ─────────────────────────────────────────────────────────────────────────────
def train_and_eval_fold(model, x, graph_repr, y, train_mask, val_mask,
                        epochs=150, lr=0.01, weight_decay=5e-4, device='cpu'):
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    if isinstance(graph_repr, torch.Tensor):
        graph_repr = graph_repr.to(device)

    # Xử lý class weight
    y_train = y[train_mask]
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    weight = torch.tensor([n_pos / max(n_neg, 1), 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    best_val_auc = 0.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, graph_repr)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(x, graph_repr)
            probs = F.softmax(out_eval[val_mask], dim=1)[:, 1].cpu().numpy()
            preds = out_eval[val_mask].argmax(dim=1).cpu().numpy()
            y_val = y[val_mask].cpu().numpy()

            val_acc = accuracy_score(y_val, preds)
            val_bal_acc = balanced_accuracy_score(y_val, preds)
            try:
                val_auc = roc_auc_score(y_val, probs)
            except ValueError:
                val_auc = 0.5
            val_f1 = f1_score(y_val, preds, zero_division=0)

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            best_metrics = {
                'acc': val_acc,
                'bal_acc': val_bal_acc,
                'auc': val_auc,
                'f1': val_f1,
                'sens': sens,
                'spec': spec
            }

    return best_metrics


def run_cross_validation(features_raw, pheno_df, labels, n_splits=10, k_best=2000,
                         model_type='gcn', epochs=150, lr=0.01, device='cpu'):
    """
    Chạy Stratified Cross-Validation với Fold-wise Feature Selection (chống leakage).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {
        'acc': [], 'bal_acc': [], 'auc': [], 'f1': [], 'sens': [], 'spec': []
    }

    # Xây dựng ma trận kề dân số W
    adj_matrix = compute_population_adjacency(features_raw, pheno_df, k_nn=10, age_threshold=2.0)
    adj_norm = normalize_adjacency_matrix(adj_matrix)
    adj_norm_tensor = torch.FloatTensor(adj_norm)

    # Chuyển đổi thành edge_index cho PyG nếu cần
    if HAS_PYG and model_type in ['gat', 'pyg_gcn']:
        rows, cols = np.where(adj_matrix > 0)
        edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
        edge_weight = torch.tensor(adj_matrix[rows, cols], dtype=torch.float)
        graph_repr = edge_index
    else:
        graph_repr = adj_norm_tensor

    y_tensor = torch.LongTensor(labels)

    print(f"\n🚀 Bắt đầu {n_splits}-Fold Stratified Cross-Validation [{model_type.upper()}]...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(features_raw, labels)):
        # ── Fold-wise Feature Selection (Tránh 100% Data Leakage) ──
        selector = SelectKBest(f_classif, k=min(k_best, features_raw.shape[1]))
        scaler = StandardScaler()

        # Fit selector ONLY on train_idx
        X_train_sel = selector.fit_transform(features_raw[train_idx], labels[train_idx])
        X_train_norm = scaler.fit_transform(X_train_sel)

        # Transform all nodes using train-fitted parameters
        X_all_sel = selector.transform(features_raw)
        X_all_norm = scaler.transform(X_all_sel)
        x_tensor = torch.FloatTensor(X_all_norm)

        # Tạo mask
        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        val_mask = torch.zeros(len(labels), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        # Khởi tạo mô hình
        if model_type == 'gat':
            model = PopulationGAT(in_features=k_best, hidden_dim1=32, num_heads=4, hidden_dim2=32)
        else:
            model = PopulationGCN(in_features=k_best, hidden_dim1=64, hidden_dim2=32)

        # Huấn luyện fold
        fold_res = train_and_eval_fold(
            model, x_tensor, graph_repr, y_tensor, train_mask, val_mask,
            epochs=epochs, lr=lr, device=device
        )

        for metric in results:
            results[metric].append(fold_res[metric])

        print(f"  Fold {fold+1:2d}/{n_splits} | Acc: {fold_res['acc']*100:.1f}% | "
              f"BalAcc: {fold_res['bal_acc']*100:.1f}% | AUC: {fold_res['auc']*100:.1f}% | "
              f"F1: {fold_res['f1']*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"📊 KẾT QUẢ TỔNG HỢP {n_splits}-FOLD CV [{model_type.upper()}]")
    print(f"{'='*60}")
    for metric, values in results.items():
        print(f"  {metric.upper():<12}: {np.mean(values)*100:.2f}% ± {np.std(values)*100:.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hàm Tải Dữ Liệu
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_dir='data', npz_path=None):
    """
    Tải dữ liệu ma trận FC và thông tin phenotypic từ file nội bộ.
    """
    # Nếu có file npz sẵn
    if npz_path and os.path.exists(npz_path):
        print(f"📂 Đang tải dữ liệu từ {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        fc_matrices = data['fc_matrices']
        y = data['y']
        pheno_path = os.path.join(data_dir, 'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv')
        df_pheno = pd.read_csv(pheno_path)
    else:
        # Tải từ danh sách 1D và phenotypic
        print("📂 Đang quét các file ROI 1D trong thư mục cpac/filt_noglobal...")
        files = sorted(glob.glob(os.path.join(data_dir, 'ABIDE_pcp', 'cpac', 'filt_noglobal', '*cc200*.1D')))
        pheno_path = os.path.join(data_dir, 'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv')
        df_pheno_raw = pd.read_csv(pheno_path)

        # Lấy file_id từ tên file
        valid_sub_ids = []
        valid_files = []
        for f in files:
            parts = os.path.basename(f).split('_')
            sub_id = int(parts[1]) if len(parts) >= 2 else None
            if sub_id and sub_id in df_pheno_raw['SUB_ID'].values:
                valid_sub_ids.append(sub_id)
                valid_files.append(f)

        print(f"✅ Tìm thấy {len(valid_files)} subjects hợp lệ khớp giữa file 1D và phenotypic data.")
        df_pheno = df_pheno_raw[df_pheno_raw['SUB_ID'].isin(valid_sub_ids)].copy()
        df_pheno.sort_values('SUB_ID', inplace=True)
        df_pheno.reset_index(drop=True, inplace=True)

        # Tính ma trận FC cho các subject
        triu_idx = np.triu_indices(200, k=1)
        features_list = []
        labels_list = []

        print("⚙️ Đang tính ma trận tương quan Pearson FC cho từng subject...")
        for _, row in df_pheno.iterrows():
            sub_id = row['SUB_ID']
            match_file = [f for f in valid_files if f"_{sub_id:07d}_" in f or f"_{sub_id}_" in f][0]
            ts = np.loadtxt(match_file)
            if ts.shape[1] != 200:
                continue
            corr = np.corrcoef(ts.T)
            np.nan_to_num(corr, copy=False, nan=0.0)
            fc_vec = corr[triu_idx]
            features_list.append(fc_vec)
            # DX_GROUP: 1=ASD, 2=TD
            labels_list.append(1 if row['DX_GROUP'] == 1 else 0)

        features = np.array(features_list)
        y = np.array(labels_list)
        return features, df_pheno, y

    # Nếu dùng npz, trích xuất tam giác trên
    triu_idx = np.triu_indices(fc_matrices.shape[1], k=1)
    features = np.array([fc[triu_idx] for fc in fc_matrices])
    return features, df_pheno, y


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Population GCN for ASD Classification")
    parser.add_argument('--data_dir', type=str, default='data', help="Thư mục chứa dữ liệu ABIDE")
    parser.add_argument('--npz_path', type=str, default=None, help="Đường dẫn file npz nếu có")
    parser.add_argument('--n_splits', type=int, default=10, help="Số fold cross-validation (10 hoặc 5)")
    parser.add_argument('--k_best', type=int, default=2000, help="Số feature FC chọn lọc (chuẩn Parisot: 2000)")
    parser.add_argument('--epochs', type=int, default=150, help="Số epoch huấn luyện mỗi fold")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat'], help="Kiến trúc mạng (gcn/gat)")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Thiết bị sử dụng: {device}")

    features, df_pheno, y = load_data(args.data_dir, args.npz_path)
    print(f"✅ Dữ liệu sẵn sàng: {features.shape[0]} subjects, {features.shape[1]} raw FC features mỗi subject.")

    # Chạy mô hình
    results = run_cross_validation(
        features_raw=features,
        pheno_df=df_pheno,
        labels=y,
        n_splits=args.n_splits,
        k_best=args.k_best,
        model_type=args.model,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )
