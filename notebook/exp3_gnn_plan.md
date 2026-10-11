# 🕸️ Exp 3 — Graph Neural Network cho ASD Classification
## Implementation Plan: GCN → GAT trên ABIDE I (CC200 Atlas)

---

## Tổng quan kiến trúc dữ liệu

```
FC Matrix (200×200)
       ↓ Graph Construction
Graph G = (V, E, X)
  V: 200 nodes (ROIs)
  E: edges khi |FC[i,j]| > threshold
  X: node features = row FC matrix (200-dim mỗi node)
       ↓
GCN / GAT layers
       ↓
Global Pooling → Graph-level vector
       ↓
MLP Classifier → P(ASD)
```

---

## Step 1 — Môi trường & Thư viện

```bash
pip install torch torchvision
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install nilearn scikit-learn numpy pandas matplotlib seaborn
```

```python
# Kiểm tra version
import torch
import torch_geometric
print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

---

## Step 2 — Data Pipeline: FC Matrix → PyG Graph Objects

```python
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

def fc_to_graph(fc_matrix, label, threshold=0.3):
    """
    Chuyển FC matrix (200×200) thành PyTorch Geometric Data object.
    
    Args:
        fc_matrix: numpy array (200, 200) - ma trận tương quan Pearson
        label:     int (0=TD, 1=ASD)
        threshold: float - ngưỡng để tạo cạnh đồ thị
    
    Returns:
        PyG Data object với:
          - x:         node features (200, 200)
          - edge_index: danh sách cạnh (2, num_edges)
          - edge_attr:  trọng số cạnh (num_edges,)
          - y:         nhãn phân loại
    """
    n_rois = fc_matrix.shape[0]   # = 200
    
    # ── Node Features: mỗi node dùng hàng tương ứng trong FC matrix ──
    # Shape: (200, 200) — node i biết nó kết nối mạnh với những node nào
    x = torch.tensor(fc_matrix, dtype=torch.float)
    
    # ── Edge Construction: chỉ tạo cạnh khi |FC| > threshold ──
    rows, cols = np.where(np.abs(fc_matrix) > threshold)
    
    # Loại bỏ self-loops (đường chéo)
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]
    
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    edge_attr  = torch.tensor(fc_matrix[rows, cols],  dtype=torch.float)
    
    # ── Label ──
    y = torch.tensor([label], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ── Tạo dataset từ fc_matrices và y đã có ở notebook ──
def build_graph_dataset(fc_matrices, labels, threshold=0.3):
    """
    fc_matrices: numpy array (n_subjects, 200, 200)
    labels:      numpy array (n_subjects,)
    """
    graph_list = []
    for i in range(len(labels)):
        g = fc_to_graph(fc_matrices[i], labels[i], threshold)
        graph_list.append(g)
    return graph_list


# Tạo dataset
graph_dataset = build_graph_dataset(fc_matrices, y, threshold=0.3)

# Kiểm tra 1 sample
g = graph_dataset[0]
print(f"Nodes: {g.num_nodes}")            # 200
print(f"Edges: {g.num_edges}")            # tùy threshold
print(f"Node features: {g.x.shape}")     # (200, 200)
print(f"Label: {g.y.item()}")            # 0 hoặc 1
print(f"Edge density: {g.num_edges / (200*199):.1%}")
```

---

## Step 3 — Phân tích Graph Properties (Trước khi train)

```python
import matplotlib.pyplot as plt

# Phân tích ảnh hưởng của threshold đến số cạnh
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
avg_edges  = []

for thr in thresholds:
    edges_list = []
    for fc in fc_matrices:
        n_edges = np.sum(np.abs(fc) > thr) - 200  # trừ self-loops
        edges_list.append(n_edges)
    avg_edges.append(np.mean(edges_list))
    print(f"Threshold={thr}: avg edges={np.mean(edges_list):.0f} "
          f"({np.mean(edges_list)/(200*199):.1%} density)")

# Chọn threshold: đủ sparse để tránh overfit, đủ dense để giữ signal
# Khuyến nghị: threshold=0.3 (density ~30-40%)
```

---

## Step 4 — Kiến trúc Model

### Option A: GCN (Graph Convolutional Network)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class BrainGCN(nn.Module):
    """
    GCN cho brain connectivity classification.
    Kiến trúc: 3 lớp GCN → Global Pooling → MLP
    """
    def __init__(self, in_channels=200, hidden_dim=64, num_classes=2, dropout=0.5):
        super(BrainGCN, self).__init__()
        
        # ── GCN Layers ──
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # ── Batch Normalization ──
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # ── Classifier ──
        # Concat mean + max pooling → 2 * (hidden_dim // 2)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, edge_index, batch):
        # ── Graph Convolution ──
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)
        
        # ── Readout: Mean + Max Pooling (concat) ──
        h_mean = global_mean_pool(h, batch)   # (batch_size, hidden//2)
        h_max  = global_max_pool(h, batch)    # (batch_size, hidden//2)
        h_graph = torch.cat([h_mean, h_max], dim=1)  # (batch_size, hidden)
        
        # ── MLP Classification ──
        h_graph = self.dropout(h_graph)
        h_graph = F.relu(self.fc1(h_graph))
        h_graph = self.dropout(h_graph)
        out = self.fc2(h_graph)
        
        return out
```

### Option B: GAT (Graph Attention Network — Mạnh hơn)

```python
from torch_geometric.nn import GATConv

class BrainGAT(nn.Module):
    """
    GAT với multi-head attention — học ROI nào quan trọng hơn.
    Advantage: Attention weights có thể visualize → giải thích được.
    """
    def __init__(self, in_channels=200, hidden_dim=32, 
                 num_heads=4, num_classes=2, dropout=0.5):
        super(BrainGAT, self).__init__()
        
        # ── GAT Layers ──
        # Layer 1: concat multi-head → hidden_dim * num_heads output
        self.gat1 = GATConv(in_channels, hidden_dim, heads=num_heads,
                             dropout=dropout, concat=True)
        # Layer 2: average multi-head → hidden_dim output  
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1,
                             dropout=dropout, concat=False)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # ── Classifier ──
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, edge_index, batch):
        # ── GAT Layers ──
        h = self.gat1(x, edge_index)   # (n_nodes, hidden*heads)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.dropout(h)
        
        h = self.gat2(h, edge_index)   # (n_nodes, hidden)
        h = self.bn2(h)
        h = F.elu(h)
        
        # ── Readout ──
        h_mean = global_mean_pool(h, batch)
        h_max  = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=1)
        
        # ── Classifier ──
        h_graph = self.dropout(h_graph)
        h_graph = F.elu(self.fc1(h_graph))
        h_graph = self.dropout(h_graph)
        out = self.fc2(h_graph)
        
        return out
    
    def get_attention_weights(self, x, edge_index):
        """Trả về attention weights để visualize kết nối quan trọng."""
        _, (edge_index_out, alpha) = self.gat1(
            x, edge_index, return_attention_weights=True
        )
        return edge_index_out, alpha
```

---

## Step 5 — Training Loop

```python
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
import numpy as np

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    for batch in loader:
        batch = batch.to(device)
        out  = model(batch.x, batch.edge_index, batch.batch)
        prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        pred = out.argmax(dim=1).cpu().numpy()
        
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred)
        y_prob.extend(prob)
    
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc     = roc_auc_score(y_true, y_prob)
    f1      = f1_score(y_true, y_pred)
    
    return bal_acc, auc, f1


def run_cv_gnn(graph_dataset, y, n_splits=5, 
               model_class=BrainGCN, epochs=100,
               lr=1e-3, weight_decay=1e-4,
               batch_size=32, device='cpu'):
    """
    5-Fold Stratified CV cho GNN.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = {
        'val_bal_acc': [], 'val_auc': [], 'val_f1': [],
        'train_bal_acc': [], 'train_auc': [], 'train_f1': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"\n{'─'*50}")
        print(f"  Fold {fold+1}/{n_splits}")
        print(f"{'─'*50}")
        
        # ── Data Split ──
        train_data = [graph_dataset[i] for i in train_idx]
        val_data   = [graph_dataset[i] for i in val_idx]
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)
        
        # ── Model & Optimizer ──
        model     = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        
        # Class weights để xử lý imbalance
        y_train = y[train_idx]
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
        criterion = nn.CrossEntropyLoss()
        
        # ── Training Loop ──
        best_val_auc = 0
        best_state   = None
        
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            
            if epoch % 20 == 0:
                val_bal_acc, val_auc, val_f1 = evaluate(model, val_loader, device)
                print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | "
                      f"Val BalAcc: {val_bal_acc:.3f} | AUC: {val_auc:.3f}")
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        
        # ── Evaluate Best Model ──
        model.load_state_dict(best_state)
        train_metrics = evaluate(model, train_loader, device)
        val_metrics   = evaluate(model, val_loader,   device)
        
        fold_results['train_bal_acc'].append(train_metrics[0])
        fold_results['train_auc'].append(train_metrics[1])
        fold_results['train_f1'].append(train_metrics[2])
        fold_results['val_bal_acc'].append(val_metrics[0])
        fold_results['val_auc'].append(val_metrics[1])
        fold_results['val_f1'].append(val_metrics[2])
        
        print(f"\n  ✅ Best → Val BalAcc: {val_metrics[0]:.3f} | AUC: {val_metrics[1]:.3f}")
    
    return fold_results
```

---

## Step 6 — Chạy thực nghiệm & So sánh

```python
# ── Config ──
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.3
EPOCHS    = 100

print(f"Device: {DEVICE}")
print(f"Graph threshold: {THRESHOLD}")

# ── Tạo dataset ──
graph_dataset = build_graph_dataset(fc_matrices, y, threshold=THRESHOLD)

# ── Exp 3a: GCN ──
print("\n" + "="*55)
print("   EXP 3a — BrainGCN")
print("="*55)
results_gcn = run_cv_gnn(
    graph_dataset, y,
    model_class=BrainGCN,
    epochs=EPOCHS,
    device=DEVICE
)

# ── Exp 3b: GAT ──
print("\n" + "="*55)
print("   EXP 3b — BrainGAT")
print("="*55)
results_gat = run_cv_gnn(
    graph_dataset, y,
    model_class=BrainGAT,
    epochs=EPOCHS,
    device=DEVICE
)

# ── Tổng hợp kết quả ──
print("\n" + "="*65)
print("   TỔNG KẾT QUẢ — GNN vs SVM Baseline")
print("="*65)

comparison = {
    'SVM Baseline':  {'BalAcc': 64.5, 'AUC': 70.0, 'F1': 60.7},
    'GCN (Exp 3a)':  {
        'BalAcc': np.mean(results_gcn['val_bal_acc']) * 100,
        'AUC':    np.mean(results_gcn['val_auc']) * 100,
        'F1':     np.mean(results_gcn['val_f1']) * 100
    },
    'GAT (Exp 3b)':  {
        'BalAcc': np.mean(results_gat['val_bal_acc']) * 100,
        'AUC':    np.mean(results_gat['val_auc']) * 100,
        'F1':     np.mean(results_gat['val_f1']) * 100
    },
}

for model_name, metrics in comparison.items():
    print(f"\n  {model_name}:")
    for metric, val in metrics.items():
        print(f"    {metric:<12}: {val:.1f}%")
```

---

## Step 7 — Ablation Study: Ảnh hưởng của Threshold

```python
thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5]
ablation_results   = {}

for thr in thresholds_to_test:
    print(f"\n--- Threshold = {thr} ---")
    dataset_thr = build_graph_dataset(fc_matrices, y, threshold=thr)
    res = run_cv_gnn(dataset_thr, y, model_class=BrainGAT,
                     epochs=80, device=DEVICE)
    ablation_results[thr] = np.mean(res['val_bal_acc']) * 100
    print(f"  BalAcc = {ablation_results[thr]:.1f}%")

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(list(ablation_results.keys()), 
         list(ablation_results.values()), 'bo-', linewidth=2, markersize=8)
plt.axhline(64.5, color='red', linestyle='--', label='SVM Baseline (64.5%)')
plt.xlabel('FC Threshold', fontsize=12)
plt.ylabel('Balanced Accuracy (%)', fontsize=12)
plt.title('Ablation: Graph Threshold vs Performance')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('exp3_ablation_threshold.png', dpi=120)
plt.show()
```

---

## Step 8 — Visualization: Attention Weights (Chỉ dành cho GAT)

```python
from nilearn import plotting
import numpy as np

def visualize_attention(model, graph_dataset, idx=0, top_k=20):
    """
    Hiển thị top-k kết nối não quan trọng nhất theo GAT attention.
    """
    model.eval()
    g = graph_dataset[idx].to('cpu')
    
    with torch.no_grad():
        edge_idx, alpha = model.get_attention_weights(g.x, g.edge_index)
    
    # Lấy trung bình attention qua các heads
    alpha_mean = alpha.mean(dim=1).numpy()
    
    # Top-k edges
    top_k_idx = np.argsort(alpha_mean)[-top_k:]
    src_nodes  = edge_idx[0, top_k_idx].numpy()
    dst_nodes  = edge_idx[1, top_k_idx].numpy()
    weights    = alpha_mean[top_k_idx]
    
    # Tạo connectivity matrix từ attention weights
    attention_matrix = np.zeros((200, 200))
    for s, d, w in zip(src_nodes, dst_nodes, weights):
        attention_matrix[s, d] = w
        attention_matrix[d, s] = w
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_matrix, cmap='hot', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(f'Top-{top_k} Connections theo GAT Attention\n'
              f'(Subject: {"ASD" if g.y.item() == 1 else "TD"})')
    plt.xlabel('ROI Index')
    plt.ylabel('ROI Index')
    plt.tight_layout()
    plt.savefig('exp3_attention_map.png', dpi=120)
    plt.show()

# Visualize cho 1 subject ASD và 1 subject TD
asd_idx = np.where(y == 1)[0][0]
td_idx  = np.where(y == 0)[0][0]
visualize_attention(model_gat, graph_dataset, idx=asd_idx, top_k=20)
```

---

## Cấu trúc thư mục đề xuất

```
asd-classification/
├── notebook/
│   └── ASD_SVM_Pipeline.ipynb          ← Phase 1 (đã xong)
├── exp3_gnn/
│   ├── 01_data_preparation.ipynb       ← Step 2, 3
│   ├── 02_model_gcn.ipynb              ← Step 4a, 5, 6
│   ├── 03_model_gat.ipynb              ← Step 4b, 5, 6
│   ├── 04_ablation_threshold.ipynb     ← Step 7
│   ├── 05_visualization.ipynb          ← Step 8
│   └── models/
│       ├── brain_gcn.py                ← BrainGCN class
│       └── brain_gat.py                ← BrainGAT class
└── data/
    └── ABIDE_pcp/                      ← Data đã tải
```

---

## Kết quả kỳ vọng & Baseline comparison

| Model | BalAcc | AUC-ROC | F1 | Ghi chú |
|---|---|---|---|---|
| SVM (Phase 1) | ~65% | ~70% | ~61% | Baseline |
| GCN (Exp 3a)  | ~67% | ~71% | ~63% | +2% nhờ graph structure |
| GAT (Exp 3b)  | ~68–72% | ~72–75% | ~64–67% | +attention mechanism |

> **Lưu ý:** Nếu kết quả GAT < GCN, kiểm tra lại threshold và dropout.  
> GAT mạnh hơn về mặt lý thuyết nhưng cần tune nhiều hơn.
