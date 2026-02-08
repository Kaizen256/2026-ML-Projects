import os, json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

SEED = 6
np.random.seed(SEED)

MEANS_PATH = "data/training_data_means.csv"
VALMAP_PATH = "data/pert_ids_val.csv"

N_PATH = os.path.join("external_genept", "GenePT_emebdding_v2", "NCBI_summary_of_genes.json")
U_PATH = os.path.join("external_genept", "GenePT_emebdding_v2", "NCBI_UniProt_summary_of_genes.json")

OUT_N = os.path.join("external_genept", "tfidf_ncbi_svd128.npz")
OUT_U = os.path.join("external_genept", "tfidf_uniprot_svd128.npz")
OUT_C = os.path.join("external_genept", "tfidf_combined_svd128.npz")

SVD_DIM = 128

df_means = pd.read_csv(MEANS_PATH)
gene_cols = [c for c in df_means.columns if c != "pert_symbol"]

base_mask = df_means["pert_symbol"].astype(str) == "non-targeting"
df_train = df_means.loc[~base_mask].reset_index(drop=True)
train_genes = df_train["pert_symbol"].astype(str).tolist()

df_valmap = pd.read_csv(VALMAP_PATH)
val_map = dict(zip(df_valmap["pert_id"].astype(str), df_valmap["pert"].astype(str)))
val_genes = list(val_map.values())

union_genes = sorted(set(gene_cols) | set(train_genes) | set(val_genes))
union_upper = [g.upper() for g in union_genes]

print("Union genes:", len(union_genes))

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

ncbi = load_json(N_PATH)
unip = load_json(U_PATH)

def build_text_list(d, genes_upper, default=""):
    texts = []
    missing = 0
    for g in genes_upper:
        t = d.get(g, None)
        if t is None or not str(t).strip():
            texts.append(default)
            missing += 1
        else:
            texts.append(str(t))
    return texts, missing

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=200_000,
)

def fit_embed(texts, out_path, name):
    X = vectorizer.fit_transform(texts)
    print(f"[{name}] TFIDF shape:", X.shape)

    k = min(SVD_DIM, X.shape[0] - 1, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    E = svd.fit_transform(X).astype(np.float32)

    # if k < SVD_DIM, pad so downstream code is consistent
    if E.shape[1] < SVD_DIM:
        pad = np.zeros((E.shape[0], SVD_DIM - E.shape[1]), dtype=np.float32)
        E = np.hstack([E, pad])

    np.savez_compressed(out_path, genes=np.array(union_upper, dtype=object), emb=E)
    print(f"[{name}] wrote {out_path} emb:", E.shape, "explained@k:", float(svd.explained_variance_ratio_.sum()))
    return E

# NCBI only
texts_n, miss_n = build_text_list(ncbi, union_upper, default="")
print("NCBI missing summaries:", miss_n)
E_n = fit_embed(texts_n, OUT_N, "NCBI")

# UniProt enriched
texts_u, miss_u = build_text_list(unip, union_upper, default="")
print("UniProt missing summaries:", miss_u)
E_u = fit_embed(texts_u, OUT_U, "UniProt")

# Combined text
texts_c = []
for tn, tu in zip(texts_n, texts_u):
    texts_c.append((tn + " " + tu).strip())
E_c = fit_embed(texts_c, OUT_C, "Combined")