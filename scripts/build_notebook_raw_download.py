"""Script to generate Notebook: Site-Specific Full ABIDE I Raw fMRI Preprocessed Dataset Downloader for Kaggle.

This script constructs notebooks/01_abide_raw_download.ipynb adhering to the specifications
in GitHub Issue #18.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def create_markdown_cell(source_lines: List[str]) -> Dict[str, Any]:
    """Construct an nbformat-compliant markdown cell dictionary.

    Args:
        source_lines: Sequence of strings representing lines of markdown content.

    Returns:
        Dictionary conforming to nbformat version 4 markdown cell schema.
    """
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]],
    }


def create_code_cell(source_lines: List[str]) -> Dict[str, Any]:
    """Construct an nbformat-compliant code cell dictionary.

    Args:
        source_lines: Sequence of strings representing lines of Python executable code.

    Returns:
        Dictionary conforming to nbformat version 4 code cell schema.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]],
    }


def build_notebook() -> None:
    """Generate and write the standalone ABIDE I raw download Kaggle notebook.

    Assembles markdown explanations and self-contained code cells into
    notebooks/01_abide_raw_download.ipynb conforming to nbformat v4.
    """
    cells = []

    # ---------------------------------------------------------
    # Cell 1: Header and Overview
    # ---------------------------------------------------------
    cells.append(
        create_markdown_cell([
            "# ABIDE I Site-Specific Raw fMRI Preprocessed Dataset Downloader for Kaggle",
            "",
            "## 1. Problem Context and Architectural Design",
            "",
            "### Kaggle Storage Envelope (20 GB Limit)",
            "The Autism Brain Imaging Data Exchange I (ABIDE I) cohort comprises 1,112 subjects across international neuroimaging centers. In its preprocessed 4D functional Magnetic Resonance Imaging (rs-fMRI) NIfTI format (`*_func_preproc.nii.gz`), each scan averages ~97 MB, totaling ~105 GB across all subjects.",
            "",
            "Because Kaggle enforces a strict **20 GB limit** on `/kaggle/working`, downloading the full repository or even half of it triggers immediate disk exhaustion and kernel termination.",
            "",
            "### Per-Institution Modular Partitioning",
            "This downloader consolidates the 1,112 subjects from 20 distinct site identifiers into **17 primary institutions**, automatically merging paired sub-sites (`UM_1`/`UM_2` into `UM`, `UCLA_1`/`UCLA_2` into `UCLA`, and `LEUVEN_1`/`LEUVEN_2` into `LEUVEN`).",
            "",
            "Because the largest single institution (**NYU** with 184 subjects) consumes approximately **18.0 GB**, every single institution execution strictly respects the 20 GB `/kaggle/working` limit without needing mid-run file deletion or 2D slice downsampling.",
            "",
            "### Preserving Raw 4D Volumetric NIfTI Format",
            "Unlike multi-view 2D slice projection pipelines, this notebook preserves intact 4D volumetric NIfTI acquisitions (`.nii.gz`). This enables downstream training of custom 3D/4D Convolutional Neural Networks, Spatio-Temporal Graph Neural Networks, or topological time-series analysis.",
            "",
            "### Self-Contained Kaggle Dataset Artifact Structure",
            "Each execution produces a self-contained, publication-ready dataset folder:",
            "```text",
            "/kaggle/working/abide_i_{institution.lower()}_raw/",
            "|-- metadata.csv              # Full phenotypic manifest with covariates and download status",
            "|-- dataset-metadata.json      # Kaggle Dataset descriptor for one-click publishing",
            "`-- scans/                    # Raw 4D preprocessed fMRI NIfTI files",
            "    `-- {file_id}_func_preproc.nii.gz",
            "```",
        ])
    )

    # ---------------------------------------------------------
    # Cell 2: Environment and Dependencies
    # ---------------------------------------------------------
    cells.append(
        create_markdown_cell([
            "## 2. Environment Setup and Library Imports",
            "",
            "We import required standard libraries and scientific packages.",
        ])
    )

    cells.append(
        create_code_cell([
            "import os",
            "import sys",
            "import time",
            "import json",
            "import shutil",
            "import urllib.request",
            "import urllib.error",
            "from pathlib import Path",
            "from typing import Dict, List, Optional, Tuple, Any",
            "",
            "import pandas as pd",
            "import numpy as np",
            "",
            "print('[INFO] Standard libraries and scientific stack loaded successfully.')",
        ])
    )

    # ---------------------------------------------------------
    # Cell 3: Canonical Institution Registry & Mapping
    # ---------------------------------------------------------
    cells.append(
        create_markdown_cell([
            "## 3. Institution Consolidation Registry",
            "",
            "We define the canonical registry of the 17 institutions, mapping the 20 ABIDE site identifiers into primary institutions.",
        ])
    )

    cells.append(
        create_code_cell([
            "# Canonical reference of the 17 institutions across all 1,112 ABIDE I subjects",
            "INSTITUTION_METRICS: Dict[str, Dict[str, Any]] = {",
            "    'CALTECH':  {'expected_n': 38,  'est_gb': 3.7},",
            "    'CMU':      {'expected_n': 27,  'est_gb': 2.6},",
            "    'KKI':      {'expected_n': 55,  'est_gb': 5.3},",
            "    'LEUVEN':   {'expected_n': 64,  'est_gb': 6.2},  # Merged: LEUVEN_1 (29) + LEUVEN_2 (35)",
            "    'MAX_MUN':  {'expected_n': 57,  'est_gb': 5.5},",
            "    'NYU':      {'expected_n': 184, 'est_gb': 18.0},",
            "    'OHSU':     {'expected_n': 28,  'est_gb': 2.7},",
            "    'OLIN':     {'expected_n': 36,  'est_gb': 3.5},",
            "    'PITT':     {'expected_n': 57,  'est_gb': 5.5},",
            "    'SBL':      {'expected_n': 30,  'est_gb': 2.9},",
            "    'SDSU':     {'expected_n': 36,  'est_gb': 3.5},",
            "    'STANFORD': {'expected_n': 40,  'est_gb': 3.9},",
            "    'TRINITY':  {'expected_n': 49,  'est_gb': 4.8},",
            "    'UCLA':     {'expected_n': 109, 'est_gb': 10.6}, # Merged: UCLA_1 (82) + UCLA_2 (27)",
            "    'UM':       {'expected_n': 145, 'est_gb': 14.1}, # Merged: UM_1 (110) + UM_2 (35)",
            "    'USM':      {'expected_n': 101, 'est_gb': 9.8},",
            "    'YALE':     {'expected_n': 56,  'est_gb': 5.4},",
            "}",
            "",
            "VALID_INSTITUTIONS: List[str] = sorted(INSTITUTION_METRICS.keys())",
            "",
            "def map_site_to_institution(site_id: str) -> str:",
            "    \"\"\"Standardize raw SITE_ID into one of the 17 primary institutions.\"\"\"",
            "    site_clean = str(site_id).strip().upper()",
            "    sub_site_map = {",
            "        'UM_1': 'UM',",
            "        'UM_2': 'UM',",
            "        'UCLA_1': 'UCLA',",
            "        'UCLA_2': 'UCLA',",
            "        'LEUVEN_1': 'LEUVEN',",
            "        'LEUVEN_2': 'LEUVEN',",
            "    }",
            "    resolved = sub_site_map.get(site_clean, site_clean)",
            "    if resolved not in VALID_INSTITUTIONS:",
            "        valid_str = ', '.join(VALID_INSTITUTIONS)",
            "        raise ValueError(f\"Unknown site '{site_id}'. Valid institutions: {valid_str}\")",
            "    return resolved",
        ])
    )

    # ---------------------------------------------------------
    # Cell 4: Top-Level Configuration Interface
    # ---------------------------------------------------------
    cells.append(
        create_markdown_cell([
            "## 4. Top-Level Configuration Interface",
            "",
            "Configure target institution and execution mode:",
            "- `TARGET_SITE`: One of the 17 institutions (case-insensitive, e.g. `'NYU'`, `'UM'`, `'UCLA'`, `'PITT'`). Sub-sites like `'UM_1'` automatically resolve to `'UM'`.",
            "- `DRY_RUN`: If `True`, probes S3 availability and calculates storage sizes without downloading full multi-gigabyte scans.",
            "- `MAX_SUBJECTS`: Optional integer (e.g. `2` or `5`) for rapid debugging or verification runs.",
        ])
    )

    cells.append(
        create_code_cell([
            "# =====================================================================",
            "# TOP-LEVEL CONFIGURATION INTERFACE",
            "# =====================================================================",
            "TARGET_SITE: str = 'NYU'       # Target institution to download (case-insensitive)",
            "DRY_RUN: bool = False          # Set True to check connectivity and sizes without downloading",
            "MAX_SUBJECTS: Optional[int] = None  # Set integer (e.g. 2) for quick test, or None for all",
            "",
            "# S3 Pipeline & Derivative Settings",
            "PIPELINE: str = 'cpac'",
            "PRIMARY_STRATEGY: str = 'nofilt_noglobal'",
            "FALLBACK_STRATEGY: str = 'filt_noglobal'",
            "DERIVATIVE: str = 'func_preproc'",
            "S3_BASE_URL: str = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative'",
            "",
            "# Resolve canonical institution and configure directory structure",
            "TARGET_INSTITUTION = map_site_to_institution(TARGET_SITE)",
            "FOLDER_NAME = f'abide_i_{TARGET_INSTITUTION.lower()}_raw'",
            "IS_KAGGLE = Path('/kaggle/working').exists()",
            "OUTPUT_ROOT = Path('/kaggle/working') / FOLDER_NAME if IS_KAGGLE else Path('./output') / FOLDER_NAME",
            "SCANS_DIR = OUTPUT_ROOT / 'scans'",
            "CACHE_DIR = (Path('/kaggle/temp') if IS_KAGGLE else Path('./temp_download')).resolve()",
            "",
            "OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)",
            "SCANS_DIR.mkdir(parents=True, exist_ok=True)",
            "CACHE_DIR.mkdir(parents=True, exist_ok=True)",
            "",
            "print(f'[INFO] Target Institution: {TARGET_INSTITUTION}')",
            "print(f'[INFO] Dry Run Mode:       {DRY_RUN}')",
            "print(f'[INFO] Max Subjects:       {MAX_SUBJECTS}')",
            "print(f'[INFO] Output Directory:   {OUTPUT_ROOT}')",
            "print(f'[INFO] Scans Directory:    {SCANS_DIR}')",
        ])
    )

    # ---------------------------------------------------------
    # Cell 5: Phenotypic Data Loading and Cohort Inspection
    # ---------------------------------------------------------
    cells.append(
        create_markdown_cell([
            "## 5. Phenotypic Data Retrieval & Target Cohort Filtering",
            "",
            "We retrieve `Phenotypic_V1_0b_preprocessed1.csv`, standardize diagnostic labels (`1` for ASD, `0` for TC), map all rows to primary institutions, and extract the cohort for `TARGET_INSTITUTION`.",
        ])
    )

    cells.append(
        create_code_cell([
            "def load_standardized_phenotypic(",
            "    csv_url: str = f'{S3_BASE_URL}/Phenotypic_V1_0b_preprocessed1.csv',",
            "    cache_dir: Optional[Path] = None,",
            ") -> pd.DataFrame:",
            "    \"\"\"Load and standardize phenotypic records from local cache or S3.\"\"\"",
            "    candidate_paths = [",
            "        Path('/kaggle/input/abide-pcp/Phenotypic_V1_0b_preprocessed1.csv'),",
            "        Path('/kaggle/input/abide-phenotypic/Phenotypic_V1_0b_preprocessed1.csv'),",
            "        Path('notebooks/data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'),",
            "        Path('notebooks/abide_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'),",
            "        Path('data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'),",
            "    ]",
            "    if cache_dir is not None:",
            "        candidate_paths.append(cache_dir / 'Phenotypic_V1_0b_preprocessed1.csv')",
            "",
            "    target_csv = None",
            "    for cand in candidate_paths:",
            "        if cand.is_file() and cand.stat().st_size > 1000:",
            "            target_csv = cand",
            "            break",
            "",
            "    if target_csv is None:",
            "        save_path = (cache_dir or Path('.')) / 'Phenotypic_V1_0b_preprocessed1.csv'",
            "        print(f'[INFO] Downloading phenotypic records from {csv_url}...')",
            "        req = urllib.request.Request(csv_url, headers={'User-Agent': 'Mozilla/5.0'})",
            "        with urllib.request.urlopen(req, timeout=60) as resp, open(save_path, 'wb') as f_out:",
            "            shutil.copyfileobj(resp, f_out)",
            "        target_csv = save_path",
            "",
            "    df = pd.read_csv(target_csv)",
            "    if len(df) != 1112:",
            "        print(f'[WARNING] Phenotypic record count is {len(df)}, expected 1,112.')",
            "",
            "    # Standardize columns",
            "    df['subject_id'] = df['SUB_ID'].astype(str)",
            "    df['site_id'] = df['SITE_ID'].astype(str)",
            "    df['institution'] = df['SITE_ID'].apply(map_site_to_institution)",
            "    df['file_id'] = df['FILE_ID'].astype(str).str.strip()",
            "    df['dx_group'] = (df['DX_GROUP'] == 1).astype(int)",
            "    df['age'] = pd.to_numeric(df['AGE_AT_SCAN'], errors='coerce')",
            "    df['sex'] = pd.to_numeric(df['SEX'], errors='coerce').fillna(0).astype(int)",
            "    df['fiq'] = pd.to_numeric(df['FIQ'], errors='coerce')",
            "    return df",
            "",
            "full_df = load_standardized_phenotypic(cache_dir=CACHE_DIR)",
            "cohort_df = full_df[full_df['institution'] == TARGET_INSTITUTION].copy().reset_index(drop=True)",
            "",
            "expected_count = INSTITUTION_METRICS[TARGET_INSTITUTION]['expected_n']",
            "est_storage_gb = INSTITUTION_METRICS[TARGET_INSTITUTION]['est_gb']",
            "",
            "print('=' * 65)",
            "print(f'COHORT SUMMARY: {TARGET_INSTITUTION}')",
            "print('=' * 65)",
            "print(f'Total Subjects:     {len(cohort_df)} (Expected: {expected_count})')",
            "print(f'Estimated Storage:  ~{est_storage_gb:.1f} GB (Kaggle Limit: 20.0 GB)')",
            "print(f'ASD Count:          {int((cohort_df[\"dx_group\"] == 1).sum())}')",
            "print(f'TC Count:           {int((cohort_df[\"dx_group\"] == 0).sum())}')",
            "print(f'Age Range:          {cohort_df[\"age\"].min():.1f} - {cohort_df[\"age\"].max():.1f} (Mean: {cohort_df[\"age\"].mean():.1f})')",
            "print(f'Unique Sub-Sites:   {list(cohort_df[\"site_id\"].unique())}')",
            "missing_file_ids = int((cohort_df['file_id'] == 'no_filename').sum())",
            "print(f'Missing File IDs:   {missing_file_ids} subjects have FILE_ID == \"no_filename\"')",
            "print('=' * 65)",
        ])
    )

    # ---------------------------------------------------------
    # Cell 6: S3 Download Engine with Exponential Backoff
    # ---------------------------------------------------------
    cells.append(
        create_markdown_cell([
            "## 6. S3 Download Engine with 3-Attempt Exponential Backoff",
            "",
            "The download engine features:",
            "1. **CPAC Multi-Strategy Query**: Primary query `nofilt_noglobal`, falling back to `filt_noglobal`.",
            "2. **Idempotency**: Detects existing local scans (> 1 MB) in `SCANS_DIR` and skips redundant network transfers.",
            "3. **3-Attempt Exponential Backoff**: Retries transient HTTP errors or timeouts using true exponential backoff ($1.0 \\times 2^{\\text{attempt}-1}$ seconds: 1s, 2s, 4s).",
            "4. **Non-Blocking Resilience**: Scans with `FILE_ID == 'no_filename'` or HTTP 404s are recorded as `missing_on_s3` in the manifest without raising unhandled exceptions or crashing the execution run.",
        ])
    )

    cells.append(
        create_code_cell([
            "def get_s3_url(",
            "    file_id: str,",
            "    pipeline: str = PIPELINE,",
            "    strategy: str = PRIMARY_STRATEGY,",
            "    derivative: str = DERIVATIVE,",
            ") -> str:",
            "    \"\"\"Construct public AWS S3 download URL for an ABIDE PCP derivative.\"\"\"",
            "    return f'{S3_BASE_URL}/Outputs/{pipeline}/{strategy}/{derivative}/{file_id}_{derivative}.nii.gz'",
            "",
            "def download_subject_scan(",
            "    file_id: str,",
            "    dest_dir: Path,",
            "    dry_run: bool = False,",
            ") -> Tuple[Optional[Path], float, str]:",
            "    \"\"\"Download single subject 4D fMRI scan with retry logic and fallback.",
            "    ",
            "    Returns:",
            "        Tuple of (local_path, file_size_mb, download_status)",
            "        download_status: 'downloaded', 'existing', 'dry_run', 'missing_on_s3', or 'failed'",
            "    \"\"\"",
            "    if not file_id or file_id.strip() == '' or file_id.strip() == 'no_filename':",
            "        return None, 0.0, 'missing_on_s3'",
            "",
            "    clean_file_id = file_id.strip()",
            "    local_filename = f'{clean_file_id}_func_preproc.nii.gz'",
            "    final_path = dest_dir / local_filename",
            "",
            "    # Idempotent check: scan already downloaded locally",
            "    if final_path.is_file() and final_path.stat().st_size > 1024 * 1024:",
            "        size_mb = round(final_path.stat().st_size / (1024 * 1024), 2)",
            "        return final_path, size_mb, 'existing'",
            "",
            "    strategies = [PRIMARY_STRATEGY]",
            "    if FALLBACK_STRATEGY not in strategies:",
            "        strategies.append(FALLBACK_STRATEGY)",
            "",
            "    # Dry-run mode: probe S3 connectivity via HEAD request without downloading content",
            "    if dry_run:",
            "        for strat in strategies:",
            "            s3_url = get_s3_url(clean_file_id, strategy=strat)",
            "            for attempt in range(1, 4):",
            "                try:",
            "                    req = urllib.request.Request(s3_url, headers={'User-Agent': 'Mozilla/5.0'}, method='HEAD')",
            "                    with urllib.request.urlopen(req, timeout=20) as resp:",
            "                        content_length_str = resp.headers.get('Content-Length')",
            "                        size_mb = round(int(content_length_str) / (1024 * 1024), 2) if content_length_str else 97.0",
            "                        return None, size_mb, 'dry_run'",
            "                except urllib.error.HTTPError as he:",
            "                    if he.code == 404:",
            "                        break  # Fall back to next strategy immediately",
            "                    time.sleep(1.0 * (2 ** (attempt - 1)))",
            "                except Exception:",
            "                    time.sleep(1.0 * (2 ** (attempt - 1)))",
            "        return None, 0.0, 'missing_on_s3'",
            "",
            "    # Full download with 3-attempt exponential backoff across strategies",
            "    temp_dest = dest_dir / f'{local_filename}.part'",
            "    last_error: Optional[Exception] = None",
            "",
            "    for strat in strategies:",
            "        s3_url = get_s3_url(clean_file_id, strategy=strat)",
            "        for attempt in range(1, 4):",
            "            try:",
            "                req = urllib.request.Request(s3_url, headers={'User-Agent': 'Mozilla/5.0'})",
            "                with urllib.request.urlopen(req, timeout=90) as resp, open(temp_dest, 'wb') as out_f:",
            "                    shutil.copyfileobj(resp, out_f)",
            "                if temp_dest.is_file() and temp_dest.stat().st_size > 1024 * 1024:",
            "                    temp_dest.rename(final_path)",
            "                    size_mb = round(final_path.stat().st_size / (1024 * 1024), 2)",
            "                    return final_path, size_mb, 'downloaded'",
            "            except urllib.error.HTTPError as he:",
            "                last_error = he",
            "                if temp_dest.is_file():",
            "                    try:",
            "                        temp_dest.unlink()",
            "                    except OSError:",
            "                        pass",
            "                if he.code == 404:",
            "                    break",
            "                time.sleep(1.0 * (2 ** (attempt - 1)))",
            "            except Exception as err:",
            "                last_error = err",
            "                if temp_dest.is_file():",
            "                    try:",
            "                        temp_dest.unlink()",
            "                    except OSError:",
            "                        pass",
            "                time.sleep(1.0 * (2 ** (attempt - 1)))",
            "",
            "    # Categorize failure without crashing execution run",
            "    if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 404:",
            "        return None, 0.0, 'missing_on_s3'",
            "    return None, 0.0, 'failed'",
        ])
    )

    # ---------------------------------------------------------
    # Cell 7: Batch Execution Loop
    # ---------------------------------------------------------
    cells.append(
        create_markdown_cell([
            "## 7. Batch Download Execution",
            "",
            "We iterate through all target cohort subjects, download each 4D fMRI NIfTI volume, track cumulative storage, and record progress into a manifest record list.",
        ])
    )

    cells.append(
        create_code_cell([
            "# Slice cohort if MAX_SUBJECTS is configured",
            "subjects_to_process = cohort_df.copy()",
            "if MAX_SUBJECTS is not None and MAX_SUBJECTS > 0:",
            "    subjects_to_process = subjects_to_process.iloc[:MAX_SUBJECTS].copy().reset_index(drop=True)",
            "    print(f'[INFO] Capped execution to first {MAX_SUBJECTS} subjects for testing.')",
            "",
            "manifest_rows = []",
            "total_bytes_downloaded = 0",
            "start_time = time.time()",
            "",
            "print(f'[INFO] Initiating batch processing for {len(subjects_to_process)} subjects...')",
            "print('-' * 80)",
            "",
            "for idx, row in subjects_to_process.iterrows():",
            "    sub_id = str(row['subject_id'])",
            "    site_id = str(row['site_id'])",
            "    institution = str(row['institution'])",
            "    file_id = str(row['file_id'])",
            "    dx_group = int(row['dx_group'])",
            "    age = row['age']",
            "    sex = int(row['sex'])",
            "    fiq = row['fiq']",
            "",
            "    scan_path, size_mb, status = download_subject_scan(",
            "        file_id=file_id,",
            "        dest_dir=SCANS_DIR,",
            "        dry_run=DRY_RUN,",
            "    )",
            "",
            "    rel_path = f'scans/{scan_path.name}' if scan_path is not None else ''",
            "    total_bytes_downloaded += int(size_mb * 1024 * 1024)",
            "",
            "    manifest_rows.append({",
            "        'subject_id': sub_id,",
            "        'site_id': site_id,",
            "        'institution': institution,",
            "        'file_id': file_id,",
            "        'dx_group': dx_group,",
            "        'age': age,",
            "        'sex': sex,",
            "        'fiq': fiq,",
            "        'file_path': rel_path,",
            "        'file_size_mb': size_mb,",
            "        'download_status': status,",
            "    })",
            "",
            "    cum_gb = total_bytes_downloaded / (1024 * 1024 * 1024)",
            "    print(",
            "        f'[{idx + 1:3d}/{len(subjects_to_process):3d}] '",
            "        f'Sub: {sub_id:<8} | File: {file_id:<20} | '",
            "        f'Status: {status:<13} | Size: {size_mb:6.1f} MB | Cum: {cum_gb:5.2f} GB'",
            "    )",
            "",
            "manifest_df = pd.DataFrame(manifest_rows)",
            "elapsed = time.time() - start_time",
            "print('-' * 80)",
            "print(f'[INFO] Batch processing finished in {elapsed / 60:.2f} minutes.')",
        ])
    )

    # ---------------------------------------------------------
    # Cell 8: Manifest and Kaggle Metadata Descriptor Export
    # ---------------------------------------------------------
    cells.append(
        create_markdown_cell([
            "## 8. Dataset Manifest Export & Kaggle Metadata Descriptor",
            "",
            "We export `metadata.csv` and `dataset-metadata.json` into `/kaggle/working/abide_i_{institution.lower()}_raw/`.",
            "This conforms to the native Kaggle Dataset specification, enabling immediate one-click publishing via UI or CLI (`kaggle datasets create -p ...`).",
        ])
    )

    cells.append(
        create_code_cell([
            "def export_dataset_artifacts(",
            "    manifest: pd.DataFrame,",
            "    output_dir: Path,",
            "    target_site: str,",
            ") -> Tuple[Path, Path]:",
            "    \"\"\"Export metadata.csv and dataset-metadata.json for native Kaggle dataset publishing.\"\"\"",
            "    site_clean = target_site.strip().upper()",
            "    site_slug = site_clean.lower()",
            "",
            "    csv_path = output_dir / 'metadata.csv'",
            "    manifest.to_csv(csv_path, index=False)",
            "    print(f'[INFO] Exported manifest to: {csv_path}')",
            "",
            "    dataset_title = f'ABIDE I {site_clean} Raw fMRI Dataset'",
            "    dataset_slug = f'abide-i-{site_slug}-raw'",
            "",
            "    kaggle_metadata = {",
            "        'title': dataset_title,",
            "        'id': f'antigravity/{dataset_slug}',",
            "        'licenses': [{'name': 'CC0-1.0'}],",
            "        'description': (",
            "            f'Preprocessed 4D resting-state fMRI NIfTI acquisitions (*_func_preproc.nii.gz) '",
            "            f'for the {site_clean} institution from the ABIDE I cohort. Preprocessed with the CPAC '",
            "            f'pipeline (nofilt_noglobal / filt_noglobal). Preserves raw 4D volumetric NIfTI format '",
            "            f'without slice downsampling, accompanied by standardized phenotypic metadata.csv.'",
            "        ),",
            "    }",
            "",
            "    json_path = output_dir / 'dataset-metadata.json'",
            "    with open(json_path, 'w', encoding='utf-8') as f:",
            "        json.dump(kaggle_metadata, f, indent=2)",
            "    print(f'[INFO] Exported Kaggle descriptor to: {json_path}')",
            "",
            "    return csv_path, json_path",
            "",
            "manifest_csv, kaggle_json = export_dataset_artifacts(",
            "    manifest=manifest_df,",
            "    output_dir=OUTPUT_ROOT,",
            "    target_site=TARGET_INSTITUTION,",
            ")",
        ])
    )

    # ---------------------------------------------------------
    # Cell 9: Verification, Summary & Publishing Instructions
    # ---------------------------------------------------------
    cells.append(
        create_markdown_cell([
            "## 9. Verification Summary & Kaggle Publishing Guide",
            "",
            "### Execution Summary",
        ])
    )

    cells.append(
        create_code_cell([
            "print('=' * 65)",
            "print(f'FINAL EXECUTION SUMMARY: {TARGET_INSTITUTION}')",
            "print('=' * 65)",
            "status_counts = manifest_df['download_status'].value_counts()",
            "for status_name, cnt in status_counts.items():",
            "    print(f'  - Status {status_name:<16}: {cnt:3d} subjects')",
            "",
            "total_size_mb = manifest_df['file_size_mb'].sum()",
            "print(f'Total Size Recorded: {total_size_mb:.1f} MB ({total_size_mb / 1024:.2f} GB)')",
            "print(f'Scans on Disk:       {len(list(SCANS_DIR.glob(\"*.nii.gz\")))} files')",
            "print('=' * 65)",
            "",
            "manifest_df.head()",
        ])
    )

    cells.append(
        create_markdown_cell([
            "### How to Publish This Dataset on Kaggle",
            "",
            "Once this notebook finishes, you can publish the resulting folder as a permanent Kaggle Dataset:",
            "",
            "#### Option A: One-Click Web UI",
            "1. In the right-hand Kaggle Notebook sidebar, locate **Output**.",
            "2. Hover over the generated folder `abide_i_{site.lower()}_raw` and click the **Create Dataset** button (three vertical dots).",
            "3. Kaggle automatically detects `dataset-metadata.json` and creates the dataset under your profile.",
            "",
            "#### Option B: Kaggle CLI (Terminal / Bash)",
            "```bash",
            "kaggle datasets create -p /kaggle/working/abide_i_{site.lower()}_raw/ -u",
            "```",
            "",
            "#### Option C: Downstream Notebook Usage",
            "In downstream model training notebooks (e.g. 3D-CNN, MIL-ResNet, or GCN), simply click **+ Add Input**, select your published dataset, and access the raw scans directly from `/kaggle/input/abide-i-{site.lower()}-raw/scans/` without waiting for repeated downloads.",
        ])
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = Path("notebooks/01_abide_raw_download.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"[SUCCESS] Built notebook with {len(cells)} cells at: {out_path.resolve()}")


if __name__ == "__main__":
    build_notebook()
