import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

# pip install pertpy
import pertpy as pt

SEED = 6
np.random.seed(SEED)

# Your files
MEANS_PATH = "data/training_data_means.csv"
OUT_DIR = "external_sig"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- load Myllia output genes ----------
df_means = pd.read_csv(MEANS_PATH)
gene_cols = [c for c in df_means.columns if c != "pert_symbol"]
myllia_genes = [g.upper() for g in gene_cols]
myllia_set = set(myllia_genes)

def _infer_pert_col(adata):
    # common names in scPerturb style datasets
    candidates = [
        "perturbation", "perturbation_name", "pert", "condition",
        "gene", "target_gene", "sgrna_target", "sgrna_symbol", "guide_target"
    ]
    obs_cols = list(adata.obs.columns)
    for c in candidates:
        if c in obs_cols:
            return c
    # fallback: try any column that looks like perturbation
    for c in obs_cols:
        cl = c.lower()
        if "pert" in cl or "target" in cl or "guide" in cl:
            return c
    raise RuntimeError(f"Could not infer perturbation column. obs cols: {obs_cols[:50]}")

def _infer_controls(labels):
    # pick controls by substring
    toks = ["non-target", "non_target", "nt", "control", "ctrl", "scramble"]
    lab = np.array(labels, dtype=str)
    keep = np.zeros(len(lab), dtype=bool)
    for t in toks:
        keep |= np.char.find(np.char.lower(lab), t) >= 0
    # if nothing matched, this will be empty and we'll handle upstream
    return np.unique(lab[keep]).tolist()

def _normalize_log2_csr(X_csr):
    # log2(x+1) on sparse CSR in-place-ish
    X = X_csr.tocsr(copy=True)
    X.data = np.log2(X.data + 1.0).astype(np.float32)
    return X

def _group_means_sparse(X_csr, groups):
    """
    X: (n_cells, n_genes) CSR
    groups: array-like length n_cells, values are strings
    Returns:
      uniq (n_groups,), means (n_groups, n_genes) CSR float32
    """
    groups = np.asarray(groups, dtype=str)
    codes, uniq = pd.factorize(groups, sort=True)
    n_groups = len(uniq)
    n_cells = X_csr.shape[0]

    # H is (n_groups, n_cells) with one 1 per cell
    H = sp.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (codes, np.arange(n_cells))),
        shape=(n_groups, n_cells)
    )
    sums = H @ X_csr  # (n_groups, n_genes)
    counts = np.bincount(codes, minlength=n_groups).astype(np.float32)
    inv = sp.diags(1.0 / np.maximum(counts, 1.0), format="csr")
    means = (inv @ sums).astype(np.float32)
    return np.array(uniq, dtype=str), means

def _clean_gene_label(x):
    # best-effort parse: "GENE", "GENE+something", "GENE|something"
    s = str(x).strip()
    for sep in ["+", "|", ",", ";", " "]:
        if sep in s:
            s = s.split(sep)[0]
    return s.upper()

def build_sig_embedding(adata, dataset_name, n_pca=128):
    pert_col = _infer_pert_col(adata)
    labels = adata.obs[pert_col].astype(str).values

    ctrl_labels = _infer_controls(labels)
    if len(ctrl_labels) == 0:
        # brute fallback: look for exact common one
        for guess in ["non-targeting", "non_targeting", "NT", "control"]:
            if guess in set(labels):
                ctrl_labels = [guess]
                break
    if len(ctrl_labels) == 0:
        raise RuntimeError(f"[{dataset_name}] Could not infer controls. Example labels: {pd.Series(labels).value_counts().head(10)}")

    # Restrict genes to Myllia outputs
    var_genes = np.array([g.upper() for g in adata.var_names])
    keep = np.array([g in myllia_set for g in var_genes], dtype=bool)
    ad = adata[:, keep].copy()

    # Normalize + log2
    sc.pp.normalize_total(ad, target_sum=1e4)
    X = ad.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = _normalize_log2_csr(X)

    # Group means
    uniq, means = _group_means_sparse(X, labels)

    # Control mean (average across all ctrl labels present)
    ctrl_mask = np.isin(uniq, np.array(ctrl_labels, dtype=str))
    ctrl_mean = means[ctrl_mask].mean(axis=0)
    ctrl_mean = np.asarray(ctrl_mean).ravel().astype(np.float32)

    # Build per-gene deltas
    deltas = []
    genes = []
    for i, lab in enumerate(uniq):
        if lab in ctrl_labels:
            continue
        g = _clean_gene_label(lab)
        # skip weird non-gene labels
        if g == "" or g.lower().startswith("non"):
            continue

        mu = np.asarray(means[i].todense()).ravel().astype(np.float32)
        deltas.append(mu - ctrl_mean)
        genes.append(g)

    if len(deltas) == 0:
        raise RuntimeError(f"[{dataset_name}] No deltas produced. Check labels/controls parsing.")

    D = np.stack(deltas, axis=0)  # (n_perts, n_genes_out)
    # PCA compress to gene signature embedding
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_pca, D.shape[0] - 1), random_state=SEED)
    E = pca.fit_transform(D).astype(np.float32)

    out_path = os.path.join(OUT_DIR, f"external_sig_{dataset_name}.npz")
    np.savez_compressed(out_path, genes=np.array(genes), emb=E, explained=pca.explained_variance_ratio_)
    print(f"[{dataset_name}] wrote {out_path} with {len(genes)} genes, emb_dim={E.shape[1]}")
    return out_path

# ---------- download and build ----------
# Smaller first: essential genes subset
adata_ess = pt.data.replogle_2022_k562_essential()  # CRISPRi K562 essential genes :contentReference[oaicite:3]{index=3}
build_sig_embedding(adata_ess, "k562_essential", n_pca=128)

# Optional bigger: genome-wide (can be heavier)
# adata_gwps = pt.data.replogle_2022_k562_gwps()     # CRISPRi K562 genome-wide :contentReference[oaicite:4]{index=4}
# build_sig_embedding(adata_gwps, "k562_gwps", n_pca=128)
