"""
retrain.py
Replika pipeline Colab Tahap 12–19
K-Means clustering + IBCF similarity + evaluasi → simpan pkl baru
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def train_kmeans(item_features, df_item, n_clusters=6, log_fn=None):
    """Tahap 12: K-Means clustering."""
    if log_fn is None: log_fn = print
    n_samples = item_features.shape[0]
    if n_clusters > n_samples:
        n_clusters = max(2, n_samples)
        log_fn(f"    ⚠️ Jumlah cluster disesuaikan ke k={n_clusters} karena produk hanya {n_samples}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_item = df_item.copy()
    df_item['cluster'] = kmeans.fit_predict(item_features)

    # Evaluasi
    labels    = df_item['cluster'].values
    sil_score = silhouette_score(item_features, labels)
    inertia   = kmeans.inertia_

    # PCA untuk visualisasi
    item_dense = item_features.toarray()
    pca        = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(item_dense)
    df_item['pca1'] = pca_result[:, 0]
    df_item['pca2'] = pca_result[:, 1]

    # Centroid PCA
    centroids_pca = (
        df_item.groupby('cluster')[['pca1','pca2']].mean()
        .sort_index().values
    )

    return kmeans, df_item, sil_score, inertia, centroids_pca


def train_ibcf(user_item_matrix):
    """Tahap 13–14: Item-item cosine similarity & predicted scores."""
    item_similarity = cosine_similarity(user_item_matrix.T)
    similarity_df   = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    numerator        = user_item_matrix.dot(similarity_df)
    denominator      = np.abs(similarity_df).sum(axis=1)
    predicted_scores = numerator.div(denominator, axis=1)

    return similarity_df, predicted_scores


def evaluate_model(test_data, predicted_scores, train_matrix, top_n=5):
    """Tahap 18–19: Evaluasi Leave-One-Out."""
    precision_list, recall_list, f1_list = [], [], []

    for _, row in test_data.iterrows():
        user_id   = row['id_user']
        true_item = row['kode_produk']

        if user_id not in train_matrix.index:
            continue
        if true_item not in predicted_scores.columns:
            continue

        user_scores = predicted_scores.loc[user_id].copy()
        purchased   = train_matrix.loc[user_id]
        purchased   = purchased[purchased > 0].index
        user_scores = user_scores.drop(
            labels=[i for i in purchased if i in user_scores.index]
        )
        if len(user_scores) == 0:
            continue

        top_items = user_scores.sort_values(ascending=False).head(top_n).index
        hit       = 1 if true_item in top_items else 0
        precision = hit / top_n
        recall    = float(hit)
        f1        = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "top_n"    : top_n,
        "precision": round(np.mean(precision_list), 4) if precision_list else 0,
        "recall"   : round(np.mean(recall_list), 4)    if recall_list    else 0,
        "f1"       : round(np.mean(f1_list), 4)        if f1_list        else 0,
        "n_users_evaluated": len(precision_list),
    }


def run_retrain(prep_result, models_dir="models", n_clusters=6, log_fn=None):
    """
    Jalankan full training dari hasil preprocessing, lalu simpan pkl.
    
    Parameters:
        prep_result (dict) : output dari preprocessing.run_preprocessing()
        models_dir (str)   : folder tujuan simpan pkl
        n_clusters (int)   : jumlah cluster K-Means
        log_fn (callable)  : fungsi logging

    Returns:
        dict berisi: silhouette, inertia, eval_results, n_items, n_users
    """
    import os
    os.makedirs(models_dir, exist_ok=True)
    if log_fn is None:
        log_fn = print

    df_item          = prep_result["df_item"]
    item_features    = prep_result["item_features"]
    user_item_matrix = prep_result["user_item_matrix"]
    train_data       = prep_result["train_data"]
    test_data        = prep_result["test_data"]

    log_fn(f"🔵 [1/4] K-Means clustering (k={n_clusters})...")
    kmeans, df_item, sil_score, inertia, centroids_pca = train_kmeans(
        item_features, df_item, n_clusters=n_clusters, log_fn=log_fn
    )
    log_fn(f"    → Silhouette: {round(sil_score, 4)} | Inertia: {round(inertia, 2)}")

    log_fn("🔵 [2/4] IBCF cosine similarity...")
    similarity_df, predicted_scores = train_ibcf(user_item_matrix)
    log_fn(f"    → Shape similarity: {similarity_df.shape}")

    log_fn("🔵 [3/4] Evaluasi model (Top-3, 5, 7)...")
    eval_results = []
    for n in [3, 5, 7]:
        res = evaluate_model(test_data, predicted_scores, user_item_matrix, top_n=n)
        eval_results.append(res)
        log_fn(f"    → Top-{n}: P={res['precision']} R={res['recall']} F1={res['f1']}")

    log_fn("💾 [4/4] Menyimpan model pkl...")
    pickle.dump(similarity_df,    open(f"{models_dir}/similarity.pkl",        'wb'))
    pickle.dump(df_item,          open(f"{models_dir}/df_item.pkl",           'wb'))
    pickle.dump(kmeans,           open(f"{models_dir}/kmeans.pkl",            'wb'))
    pickle.dump(user_item_matrix, open(f"{models_dir}/user_item_matrix.pkl",  'wb'))
    pickle.dump(centroids_pca,    open(f"{models_dir}/centroids_pca.pkl",     'wb'))
    log_fn("✅ Semua model berhasil disimpan!")

    return {
        "silhouette"  : round(sil_score, 4),
        "inertia"     : round(inertia, 2),
        "eval_results": eval_results,
        "n_items"     : len(df_item),
        "n_users"     : user_item_matrix.shape[0],
        "n_clusters"  : n_clusters,
    }