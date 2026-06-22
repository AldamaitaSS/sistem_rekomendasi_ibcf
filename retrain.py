import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def pilih_k_terbaik(k_range, sil_scores, threshold=0.01):
    k_list = list(k_range)
    best_idx = sil_scores.index(max(sil_scores))
    return k_list[best_idx]


def train_kmeans(item_features, df_item, n_clusters=None, log_fn=None, threshold=0.01):
    """Tahap 12-13: Pilih k optimal lalu fit K-Means clustering."""
    if log_fn is None: log_fn = print
    n_samples = item_features.shape[0]

    # Tentukan range k yang aman (max 10, min 2, tidak melebihi n_samples)
    k_max   = min(10, n_samples - 1)
    k_range = range(2, k_max + 1)

    if n_clusters is None:
        # Otomatis pilih k terbaik (seperti notebook)
        log_fn(f"    🔍 Menghitung silhouette untuk k=2..{k_max}...")
        log_fn(f"    📦 Jumlah produk (n_samples): {n_samples}")
        sil_scores = []
        for k in k_range:
            km_temp     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = km_temp.fit_predict(item_features)
            score       = silhouette_score(item_features, labels_temp)
            sil_scores.append(score)
            log_fn(f"    k={k} → silhouette={round(score, 4)}")

        log_fn(f"    📊 Semua skor: {[round(s,4) for s in sil_scores]}")
        best_k = pilih_k_terbaik(k_range, sil_scores)
        log_fn(f"    ✅ K optimal dipilih secara dinamis: k={best_k}")
    else:
        # Pakai k yang diminta, tapi pastikan tidak melebihi n_samples
        best_k = max(2, min(n_clusters, n_samples - 1))
        if best_k != n_clusters:
            log_fn(f"    ⚠️ Jumlah cluster disesuaikan ke k={best_k} karena produk hanya {n_samples}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_item = df_item.copy()
    df_item['cluster'] = kmeans.fit_predict(item_features)

    # Evaluasi
    labels    = df_item['cluster'].values
    sil_score = silhouette_score(item_features, labels)
    inertia   = kmeans.inertia_

    # Reduksi dimensi pakai TruncatedSVD (sparse-safe, seperti notebook baru)
    from sklearn.decomposition import TruncatedSVD
    svd        = TruncatedSVD(n_components=2, random_state=42)
    X_2d       = svd.fit_transform(item_features)
    df_item['pca1'] = X_2d[:, 0]
    df_item['pca2'] = X_2d[:, 1]

    centroids_pca = (
        df_item.groupby('cluster')[['pca1', 'pca2']].mean()
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


def evaluate_model(test_data, predicted_scores, train_matrix, top_n=3):
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


def run_retrain(prep_result, models_dir="models", n_clusters=None, threshold=0.01, log_fn=None):
    import os
    os.makedirs(models_dir, exist_ok=True)
    if log_fn is None:
        log_fn = print

    df_item          = prep_result["df_item"]
    item_features    = prep_result["item_features"]
    user_item_matrix = prep_result["user_item_matrix"]
    train_data       = prep_result["train_data"]
    test_data        = prep_result["test_data"]

    mode_txt = f"k={n_clusters}" if n_clusters else "k otomatis (dinamis)"
    log_fn(f"🔵 [1/4] K-Means clustering ({mode_txt})...")
    kmeans, df_item, sil_score, inertia, centroids_pca = train_kmeans(
        item_features, df_item, n_clusters=n_clusters, log_fn=log_fn, threshold=threshold
    )
    log_fn(f"    → K terpilih: {df_item['cluster'].nunique()} | Silhouette: {round(sil_score, 4)} | Inertia: {round(inertia, 2)}")

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
        "n_clusters"  : df_item['cluster'].nunique(),
    }