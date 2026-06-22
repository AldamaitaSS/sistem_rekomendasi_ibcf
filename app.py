from flask import Flask, render_template, request, jsonify
import os
import threading
from werkzeug.utils import secure_filename
from preprocessing import run_preprocessing, apply_user_mapping
from retrain import run_retrain
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# ── UPLOAD CONFIG ───────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT   = {"xlsx", "xls"}
retrain_status = {"running": False, "logs": [], "result": None, "error": None}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
print("Loading model...")
similarity_df    = pickle.load(open('models/similarity.pkl',        'rb'))
df_item          = pickle.load(open('models/df_item.pkl',           'rb'))
kmeans           = pickle.load(open('models/kmeans.pkl',            'rb'))
user_item_matrix = pickle.load(open('models/user_item_matrix.pkl',  'rb'))
centroids_pca    = pickle.load(open('models/centroids_pca.pkl',     'rb'))  # shape (n_clusters, 2)

# Silhouette score ASLI (dihitung dari item_features penuh saat training di notebook/retrain.py),
# BUKAN dihitung ulang dari pca1/pca2 -- karena silhouette di ruang 2D hasil reduksi dimensi
# akan berbeda nilainya dari silhouette di ruang fitur asli, dan akan salah jika ditampilkan
# sebagai "skor evaluasi sistem".
try:
    CURRENT_SILHOUETTE = pickle.load(open('models/silhouette.pkl', 'rb'))
except FileNotFoundError:
    CURRENT_SILHOUETTE = None
    print("⚠️  models/silhouette.pkl tidak ditemukan -- silhouette akan ditampilkan kosong.")
    print("    Tambahkan baris export berikut di notebook sebelum download:")
    print("    pickle.dump(sil_score, open('silhouette.pkl', 'wb'))")
print("Model loaded!")

# ── CLUSTER METADATA ───────────────────────────────────────────────────────────
# Palet & ikon disiapkan untuk k=2..10 (sesuai k_range di retrain.py),
# supaya tidak ada warna/ikon yang berulang antar cluster.
PALETTE = [
    "#f59e0b", "#8b5cf6", "#ef4444", "#10b981", "#3b82f6",
    "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
]
ICONS = ["🍪", "🍫", "🥨", "🎂", "🌶️", "🧁", "🍩", "🥖", "🍯", "🧀"]

CLUSTER_NAMES  = {}
CLUSTER_COLORS = {}
CLUSTER_ICONS  = {}

# Hitung dulu kategori dominan tiap cluster, lalu disambiguasi jika ada nama yang sama
_dominant_kategori = {}
for cid in sorted(df_item['cluster'].unique()):
    cid_int = int(cid)
    members = df_item[df_item['cluster'] == cid]
    if 'kategori' in members.columns and len(members) > 0 and members['kategori'].notna().any():
        _dominant_kategori[cid_int] = members['kategori'].value_counts().index[0]
    else:
        _dominant_kategori[cid_int] = f"Cluster {cid_int}"

# Disambiguasi: kalau ada nama yang dipakai >1 cluster, tambahkan angka urut (1), (2), dst
from collections import Counter
_name_counts = Counter(_dominant_kategori.values())
_name_seen    = {}
for cid_int, nama in _dominant_kategori.items():
    color = PALETTE[cid_int % len(PALETTE)]
    icon  = ICONS[cid_int % len(ICONS)]
    CLUSTER_COLORS[cid_int] = color
    CLUSTER_ICONS[cid_int]  = icon
    if _name_counts[nama] > 1:
        _name_seen[nama] = _name_seen.get(nama, 0) + 1
        CLUSTER_NAMES[cid_int] = f"{nama} ({_name_seen[nama]})"
    else:
        CLUSTER_NAMES[cid_int] = nama

# ── BUILD PRODUCTS LIST ────────────────────────────────────────────────────────
def build_products():
    products = []
    for idx, row in df_item.iterrows():
        kode     = row.get('kode_produk', str(idx))
        nama     = row.get('nama_produk', kode)
        kategori = row.get('kategori', '-')
        harga    = float(row.get('harga', 0))
        berat    = float(row.get('berat', 0))
        berat_g  = round(berat, 3)
        platform = str(row.get('platform', 'TikTok Shop'))
        cluster  = int(row.get('cluster', 0))
        pca1     = float(row.get('pca1', 0))
        pca2     = float(row.get('pca2', 0))
        sold     = int(user_item_matrix[kode].sum()) if kode in user_item_matrix.columns else 0
        products.append({
            "id":       kode,
            "name":     nama,
            "category": kategori,
            "price":    harga,
            "weight":   berat_g,
            "platform": platform,
            "cluster":  cluster,
            "pca1":     pca1,
            "pca2":     pca2,
            "sold":     sold,
            "in_matrix": kode in similarity_df.columns,
        })
    return products

PRODUCTS   = build_products()
CATEGORIES = sorted(set(p["category"] for p in PRODUCTS))
PLATFORMS  = sorted(set(p["platform"] for p in PRODUCTS)) or ["TikTok Shop", "Shopee"]

# ── IBCF RECOMMENDATION ────────────────────────────────────────────────────────
def get_recommendations(product_id, top_n=3):
    if product_id not in similarity_df.columns:
        prod = next((p for p in PRODUCTS if p["id"] == product_id), None)
        if not prod:
            return []
        members = [p.copy() for p in PRODUCTS if p["cluster"] == prod["cluster"] and p["id"] != product_id]
        for m in members:
            m["similarity"] = 0.0
            m["sumber"] = "Cluster"
        return members[:top_n]

    sim_scores = similarity_df[product_id].drop(index=product_id, errors='ignore')
    sim_scores = sim_scores.sort_values(ascending=False).head(top_n)
    results = []
    for kode, score in sim_scores.items():
        prod = next((p for p in PRODUCTS if p["id"] == kode), None)
        if prod:
            p = prod.copy()
            p["similarity"] = round(float(score), 4)
            p["sumber"] = "IBCF"
            results.append(p)
    return results

# ── COLD START: PREDICT CLUSTER ────────────────────────────────────────────────
# Mapping kategori → cluster dominan (dari data training)
CAT_TO_CLUSTER = {}
if 'kategori' in df_item.columns and 'cluster' in df_item.columns:
    for cid in df_item['cluster'].unique():
        members = df_item[df_item['cluster'] == cid]
        top_cat = members['kategori'].value_counts().index[0] if len(members) > 0 else ''
        CAT_TO_CLUSTER[top_cat] = int(cid)

def predict_cluster_real(kategori, harga, berat):
    n_clusters = len(CLUSTER_NAMES)

    # Utamakan kategori — paling representatif
    if kategori in CAT_TO_CLUSTER:
        cluster_id = CAT_TO_CLUSTER[kategori]
        # Buat confidence: cluster terpilih ~65%, sisanya dibagi rata
        confidences = {}
        for i in range(n_clusters):
            confidences[i] = round((100 - 65) / (n_clusters - 1), 1) if i != cluster_id else 65.0
        return cluster_id, confidences

    # Fallback: hitung jarak ke centroid K-Means pakai harga & berat ternormalisasi
    all_h = df_item['harga'].values
    all_b = df_item['berat'].values
    h_norm = (harga - all_h.min()) / (all_h.max() - all_h.min() + 1e-9)
    b_norm = (berat  - all_b.min()) / (all_b.max() - all_b.min() + 1e-9)

    centers = kmeans.cluster_centers_
    dists = [float(np.linalg.norm(np.array([h_norm, b_norm]) - c[:2])) for c in centers]
    cluster_id = int(np.argmin(dists))

    total = sum((1/(d+1e-9)) for d in dists)
    confidences = {i: round((1/(d+1e-9)) / total * 100, 1) for i, d in enumerate(dists)}
    return cluster_id, confidences

# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("rekomendasi.html", products=PRODUCTS,
        cluster_names=CLUSTER_NAMES, cluster_colors=CLUSTER_COLORS, cluster_icons=CLUSTER_ICONS)

@app.route("/rekomendasi")
def rekomendasi():
    return render_template("rekomendasi.html", products=PRODUCTS,
        cluster_names=CLUSTER_NAMES, cluster_colors=CLUSTER_COLORS, cluster_icons=CLUSTER_ICONS)

@app.route("/api/rekomendasi/<product_id>")
def api_rekomendasi(product_id):
    product = next((p for p in PRODUCTS if p["id"] == product_id), None)
    if not product:
        return jsonify({"error": "Produk tidak ditemukan"}), 404
    recs = get_recommendations(product_id)
    return jsonify({"product": product, "recommendations": recs,
        "cluster_names": CLUSTER_NAMES, "cluster_colors": CLUSTER_COLORS, "cluster_icons": CLUSTER_ICONS})

@app.route("/analisis")
def analisis():
    summary = {}
    for cid, cname in CLUSTER_NAMES.items():
        members = [p for p in PRODUCTS if p["cluster"] == cid]
        avg_price  = round(sum(p["price"]  for p in members) / len(members)) if members else 0
        avg_weight = round(sum(p["weight"] for p in members) / len(members)) if members else 0
        total_sold = sum(p["sold"] for p in members)
        # Sertakan nama produk lengkap di setiap member
        members_detail = [{"name": p["name"], "price": p["price"], "sold": p["sold"]} for p in members]
        summary[cid] = {"name": cname, "count": len(members),
            "avg_price": avg_price, "avg_weight": avg_weight,
            "total_sold": total_sold, "members": members_detail,
            "color": CLUSTER_COLORS[cid], "icon": CLUSTER_ICONS[cid]}
    return render_template("analisis.html",
        products=PRODUCTS, summary=summary,
        silhouette=round(CURRENT_SILHOUETTE, 4) if CURRENT_SILHOUETTE is not None else None,
        cluster_names=CLUSTER_NAMES, cluster_colors=CLUSTER_COLORS,
        cluster_icons=CLUSTER_ICONS)

@app.route("/coldstart")
def coldstart():
    # Redirect ke halaman clustering (cold start tidak digunakan)
    from flask import redirect, url_for
    return redirect(url_for("analisis"))

@app.route("/api/stats")
def api_stats():
    return jsonify({"n_users": int(user_item_matrix.shape[0]), "n_products": len(PRODUCTS)})

@app.route("/api/coldstart", methods=["POST"])
def api_coldstart():
    data     = request.json
    nama     = data.get("name", "")
    kategori = data.get("category", "")
    harga    = float(data.get("price", 0))
    berat    = float(data.get("weight", 0))
    platform = data.get("platform", "TikTok Shop")

    cluster, confidences = predict_cluster_real(kategori, harga, berat)
    members = [p for p in PRODUCTS if p["cluster"] == cluster]
    return jsonify({
        "name": nama, "category": kategori, "price": harga, "weight": berat, "platform": platform,
        "cluster": cluster,
        "cluster_name":  CLUSTER_NAMES.get(cluster, f"Cluster {cluster}"),
        "cluster_color": CLUSTER_COLORS.get(cluster, "#666"),
        "cluster_icon":  CLUSTER_ICONS.get(cluster, "📦"),
        "confidences": confidences,
        "silhouette_score": round(CURRENT_SILHOUETTE, 4) if CURRENT_SILHOUETTE is not None else None,
        "cluster_members": members,
        "cluster_names": CLUSTER_NAMES, "cluster_colors": CLUSTER_COLORS, "cluster_icons": CLUSTER_ICONS,
    })

@app.route("/clustering")
def clustering():
    from flask import redirect
    return redirect("/analisis")


# ── UPLOAD & RETRAIN ROUTES ────────────────────────────────────────────────

@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    global retrain_status
    if retrain_status["running"]:
        return jsonify({"error": "Proses retrain sedang berjalan, harap tunggu."}), 409

    tiktok_file = request.files.get("tiktok")
    shopee_file = request.files.get("shopee")

    if not tiktok_file and not shopee_file:
        return jsonify({"error": "Minimal 1 file harus diupload (TikTok Shop atau Shopee)."}), 400

    if tiktok_file and not allowed_file(tiktok_file.filename):
        return jsonify({"error": "Format file TikTok Shop harus .xlsx atau .xls"}), 400
    if shopee_file and not allowed_file(shopee_file.filename):
        return jsonify({"error": "Format file Shopee harus .xlsx atau .xls"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    tiktok_path, shopee_path = None, None
    if tiktok_file:
        tiktok_path = os.path.join(UPLOAD_FOLDER, secure_filename(tiktok_file.filename))
        tiktok_file.save(tiktok_path)
    if shopee_file:
        shopee_path = os.path.join(UPLOAD_FOLDER, secure_filename(shopee_file.filename))
        shopee_file.save(shopee_path)

    retrain_status = {"running": True, "logs": [], "result": None, "error": None}

    def background_retrain():
        global retrain_status, similarity_df, df_item, kmeans, user_item_matrix
        global centroids_pca, PRODUCTS, CATEGORIES, PLATFORMS
        global CLUSTER_NAMES, CLUSTER_COLORS, CLUSTER_ICONS, CURRENT_SILHOUETTE

        def log_fn(msg):
            retrain_status["logs"].append(msg)

        try:
            # Path penyimpanan data RAW gabungan (sebelum user mapping)
            old_data_path = os.path.join(UPLOAD_FOLDER, "_combined_data_raw.parquet")

            # ── STEP 1: Preprocessing data baru (tanpa user mapping) ──────────
            log_fn("⚙️ Memproses data baru...")
            prep_new = run_preprocessing(tiktok_path, shopee_path, log_fn=log_fn)
            df_new   = prep_new["df_prep"]  # masih berisi id_user asli (belum U0001 dst)

            # ── STEP 2: Gabung dengan data lama RAW ───────────────────────────
            if os.path.exists(old_data_path):
                log_fn("🔗 Menggabungkan dengan data lama...")
                df_old      = pd.read_parquet(old_data_path)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates()
                log_fn(f"    → Data lama: {len(df_old):,} baris | Data baru: {len(df_new):,} baris")
                log_fn(f"    → Total gabungan: {len(df_combined):,} baris")
            else:
                df_combined = df_new
                log_fn(f"    → Tidak ada data lama, pakai data baru saja ({len(df_combined):,} baris)")

            # ── STEP 3: Simpan RAW gabungan (sebelum mapping) ─────────────────
            # Disimpan SEBELUM mapping agar upload berikutnya bisa gabung
            # dari username asli, bukan dari U0001/U0002 yang bisa berubah
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            df_combined.to_parquet(old_data_path, index=False)
            log_fn("    → Data gabungan RAW disimpan untuk akumulasi berikutnya ✅")

            # ── STEP 4: Terapkan user ID mapping dari SELURUH data gabungan ───
            # Dilakukan setelah gabung agar username yang sama (misal 'siekomo')
            # selalu dapat ID yang sama (misal U0001) di seluruh dataset
            log_fn("🔑 Menerapkan user ID mapping dari data gabungan...")
            df_combined = apply_user_mapping(df_combined)
            log_fn(f"    → Total user unik setelah mapping: {df_combined['id_user'].nunique():,}")

            # ── STEP 5: Rebuild matrix dan features dari data gabungan ────────
            from preprocessing import (
                build_user_item_matrix, build_item_features
            )
            log_fn("📊 Membangun ulang matrix dari data gabungan...")
            user_item_matrix_new, train_data, test_data = build_user_item_matrix(df_combined, log_fn=log_fn)
            df_item_new, item_features, tfidf, encoder, scaler = build_item_features(df_combined)

            prep = {
                "df_prep": df_combined,
                "user_item_matrix": user_item_matrix_new,
                "train_data": train_data,
                "test_data": test_data,
                "df_item": df_item_new,
                "item_features": item_features,
                "tfidf": tfidf,
                "encoder": encoder,
                "scaler": scaler,
            }
            result = run_retrain(prep, models_dir="models", log_fn=log_fn)

            log_fn("🔄 Memuat ulang model ke sistem...")
            # Backup pkl lama dulu sebelum ditimpa
            import shutil, time
            backup_dir = os.path.join("models", "backup")
            os.makedirs(backup_dir, exist_ok=True)
            ts = int(time.time())
            for f_name in ["similarity.pkl","df_item.pkl","kmeans.pkl","user_item_matrix.pkl","centroids_pca.pkl"]:
                src = os.path.join("models", f_name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(backup_dir, f"{ts}_{f_name}"))

            similarity_df    = pickle.load(open("models/similarity.pkl",       "rb"))
            df_item          = pickle.load(open("models/df_item.pkl",          "rb"))
            kmeans           = pickle.load(open("models/kmeans.pkl",           "rb"))
            user_item_matrix = pickle.load(open("models/user_item_matrix.pkl", "rb"))
            centroids_pca    = pickle.load(open("models/centroids_pca.pkl",    "rb"))

            # Silhouette ASLI dari hasil training penuh (item_features), bukan dari pca1/pca2.
            # run_retrain() mengembalikan ini di dalam `result` (lihat retrain.py).
            CURRENT_SILHOUETTE = result.get("silhouette")
            pickle.dump(CURRENT_SILHOUETTE, open("models/silhouette.pkl", "wb"))

            # Re-generate metadata cluster (nama/warna/ikon) dengan logika yang SAMA
            # seperti saat startup -- termasuk disambiguasi nama yang duplikat.
            CLUSTER_NAMES.clear(); CLUSTER_COLORS.clear(); CLUSTER_ICONS.clear()
            _dominant_kategori = {}
            for cid in sorted(df_item["cluster"].unique()):
                cid_int = int(cid)
                members = df_item[df_item["cluster"] == cid]
                if "kategori" in members.columns and len(members) > 0 and members["kategori"].notna().any():
                    _dominant_kategori[cid_int] = members["kategori"].value_counts().index[0]
                else:
                    _dominant_kategori[cid_int] = f"Cluster {cid_int}"

            from collections import Counter as _Counter
            _name_counts = _Counter(_dominant_kategori.values())
            _name_seen = {}
            for cid_int, nama in _dominant_kategori.items():
                CLUSTER_COLORS[cid_int] = PALETTE[cid_int % len(PALETTE)]
                CLUSTER_ICONS[cid_int]  = ICONS[cid_int % len(ICONS)]
                if _name_counts[nama] > 1:
                    _name_seen[nama] = _name_seen.get(nama, 0) + 1
                    CLUSTER_NAMES[cid_int] = f"{nama} ({_name_seen[nama]})"
                else:
                    CLUSTER_NAMES[cid_int] = nama

            PRODUCTS[:]  = build_products()
            CATEGORIES[:] = sorted(set(p["category"] for p in PRODUCTS))
            PLATFORMS[:]  = sorted(set(p["platform"] for p in PRODUCTS)) or ["TikTok Shop","Shopee"]

            log_fn("✅ Sistem berhasil diperbarui!")
            retrain_status["result"]  = result
            retrain_status["running"] = False

        except Exception as e:
            import traceback
            retrain_status["error"]   = str(e)
            retrain_status["logs"].append(f"❌ ERROR: {e}")
            retrain_status["logs"].append(traceback.format_exc())
            retrain_status["running"] = False

    threading.Thread(target=background_retrain, daemon=True).start()
    return jsonify({"message": "Proses dimulai."})


@app.route("/api/upload/status")
def api_upload_status():
    return jsonify({
        "running": retrain_status["running"],
        "logs":    retrain_status["logs"],
        "result":  retrain_status["result"],
        "error":   retrain_status["error"],
    })

@app.route("/api/coldstart/simpan", methods=["POST"])
def api_coldstart_simpan():
    global df_item, PRODUCTS, CATEGORIES, PLATFORMS, CLUSTER_NAMES, CLUSTER_COLORS, CLUSTER_ICONS
    data     = request.json
    nama     = data.get("name", "").strip()
    kategori = data.get("category", "")
    harga    = float(data.get("price", 0))
    berat    = float(data.get("weight", 0))
    platform = data.get("platform", "")
    cluster  = int(data.get("cluster", 0))
    pca1     = float(data.get("pca1", 0))
    pca2     = float(data.get("pca2", 0))

    if not nama:
        return jsonify({"error": "Nama produk wajib diisi"}), 400

    # Buat kode produk otomatis
    prefix   = ''.join([w[0].upper() for w in nama.split()[:3]])
    existing = [p["id"] for p in PRODUCTS if p["id"].startswith(prefix)]
    kode     = f"{prefix}-NEW-{str(len(existing)+1).zfill(2)}"

    # Cek duplikat nama
    if any(p["name"].lower() == nama.lower() for p in PRODUCTS):
        return jsonify({"error": f"Produk '{nama}' sudah ada di sistem"}), 409

    # Tambah ke df_item
    new_row = {
        "kode_produk": kode, "nama_produk": nama,
        "kategori": kategori, "harga": harga,
        "berat": berat, "platform": platform,
        "cluster": cluster, "pca1": pca1, "pca2": pca2
    }
    global df_item
    import pandas as pd
    df_item = pd.concat([df_item, pd.DataFrame([new_row])], ignore_index=True)

    # Simpan pkl
    pickle.dump(df_item, open("models/df_item.pkl", "wb"))

    # Update PRODUCTS di memori
    PRODUCTS.append({
        "id": kode, "name": nama, "category": kategori,
        "price": harga, "weight": berat, "platform": platform,
        "cluster": cluster, "pca1": pca1, "pca2": pca2,
        "sold": 0, "in_matrix": False
    })
    CATEGORIES = sorted(set(p["category"] for p in PRODUCTS if p["category"]))
    PLATFORMS  = sorted(set(p["platform"] for p in PRODUCTS))

    return jsonify({"success": True, "kode": kode, "message": f"Produk '{nama}' berhasil disimpan dengan kode {kode}"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)