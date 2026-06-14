"""
preprocessing.py
Replika pipeline Colab Tahap 1–11
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

COLUMN_MAP_TIKTOK = {
    "Buyer Username"          : "id_user",
    "Seller SKU"              : "kode_produk",
    "Product Name"            : "nama_produk",
    "Product Category"        : "kategori",
    "Quantity"                : "quantity",
    "Created Time"            : "tanggal_pesanan",
    "Order Status"            : "status_pesanan",
    "SKU Unit Original Price" : "harga",
    "Weight(kg)"              : "berat",
}

COLUMN_MAP_SHOPEE = {
    "Username (Pembeli)"   : "id_user",
    "SKU Induk"            : "kode_produk",
    "Nama Produk"          : "nama_produk",
    "Jumlah"               : "quantity",
    "Waktu Pesanan Dibuat" : "tanggal_pesanan",
    "Status Pesanan"       : "status_pesanan",
    "Harga Awal"           : "harga",
    "Berat Produk"         : "berat",
}

NON_MAKANAN_KEYWORDS = [
    'plastik','kardus','kantong','bag','tas',
    'bubble','buble','wrap','packing','packaging','kotak'
]
STATUS_SELESAI = ['Completed', 'Delivered', 'Selesai']


def load_platform(file_path, column_map, platform_name):
    df = pd.read_excel(file_path, dtype=str)
    df = df.loc[:, ~df.columns.duplicated()]
    df.rename(columns=column_map, inplace=True)
    selected_cols = list(set(list(column_map.values()) + ['kategori']))
    if 'kategori' not in df.columns:
        df['kategori'] = np.nan
    cols_exist = [c for c in selected_cols if c in df.columns]
    df = df[cols_exist].copy()
    df['platform'] = platform_name
    return df


def clean_data(df_raw):
    df = df_raw.copy()
    df = df[df['status_pesanan'].isin(STATUS_SELESAI)].copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=['id_user','kode_produk','nama_produk','quantity'])
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df = df[df['quantity'] > 0].copy()
    pattern = '|'.join(NON_MAKANAN_KEYWORDS)
    df = df[~df['nama_produk'].str.lower().str.contains(pattern, na=False)].copy()
    return df


def prepare_data(df_clean):
    """
    Standarisasi format, imputasi, dan kategori.
    TIDAK melakukan user ID mapping — mapping dilakukan setelah
    data lama dan data baru digabung (lihat apply_user_mapping).
    """
    df = df_clean.copy()
    mask_shopee = df['platform'] == 'shopee'

    df.loc[mask_shopee, 'harga'] = (
        df.loc[mask_shopee, 'harga'].astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '', regex=False)
    )
    df.loc[mask_shopee, 'berat'] = (
        df.loc[mask_shopee, 'berat'].astype(str)
        .str.extract(r'(\d+\.?\d*)')[0]
        .astype(float).div(1000)
    )

    df['harga'] = pd.to_numeric(df['harga'], errors='coerce')
    df['berat'] = pd.to_numeric(df['berat'], errors='coerce')

    median_harga = (
        df[df['harga'] > 0].groupby('kode_produk')['harga'].median().to_dict()
    )
    df.loc[df['harga'] == 0, 'harga'] = (
        df.loc[df['harga'] == 0, 'kode_produk'].map(median_harga)
    )

    df['tanggal_pesanan'] = pd.to_datetime(
        df['tanggal_pesanan'].astype(str).str.strip(), errors='coerce', format='mixed'
    )
    df['kode_produk'] = df['kode_produk'].astype(str).str.strip().str.upper()

    mapping_kategori = (
        df[df['kategori'].notna()].groupby('kode_produk')['kategori'].first().to_dict()
    )
    df['kategori'] = df.apply(
        lambda row: mapping_kategori.get(row['kode_produk'], row['kategori'])
        if pd.isna(row['kategori']) else row['kategori'], axis=1
    )

    def assign_kategori(row):
        if pd.notna(row['kategori']) and str(row['kategori']).strip() != '':
            return row['kategori']
        nama = str(row['nama_produk']).lower()
        if any(k in nama for k in ['brownies', 'brownie', 'fudgy', 'bites', 'dubai', 'kunafa', 'pistachio', 'pie', 'tart']):
            return 'Kue & Pai'
        elif any(k in nama for k in ['macaron', 'macaroon', 'rice crispy', 'crumbs', 'remahan']):
            return 'Kue Camilan & Roti Pastri'
        elif any(k in nama for k in ['bola', 'bola susu', 'snack', 'camilan']):
            return 'Makanan Ringan Kering'
        elif any(k in nama for k in ['nastar', 'kukis', 'cookies', 'wisman', 'wijsman', 'sultan', 'apel', 'bundling nastar', 'paket nastar', 'premium']):
            return 'Kue Kering'
        elif any(k in nama for k in ['pia', 'meringue', 'kue busa', 'choco chips', 'chocopia', 'biskuit', 'wafer', 'lumer', 'filling', 'classic', 'homemade']):
            return 'Biskuit, Kue & Wafer'
        elif any(k in nama for k in ['sambal', 'bumbu', 'terasi', 'sachet', 'pouch', 'madura', 'bebek', 'pasta', 'saus']):
            return 'Kit Pasta & Bumbu Masak'
        elif any(k in nama for k in ['selai', 'nanas', 'olesan', 'jam', 'spread']):
            return 'Selai, Saus, & Olesan'
        else:
            return row['kategori']

    df['kategori'] = df.apply(assign_kategori, axis=1)

    # TIDAK ada user_mapping di sini — dilakukan setelah gabung
    return df


def apply_user_mapping(df):
    """
    Terapkan user ID mapping (U0001, U0002, ...) dari seluruh data gabungan.
    Harus dipanggil SETELAH concat data lama + data baru,
    sehingga username yang sama selalu dapat ID yang sama.
    """
    user_mapping = {
        user: f"U{str(i+1).zfill(4)}"
        for i, user in enumerate(df['id_user'].unique())
    }
    df = df.copy()
    df['id_user'] = df['id_user'].map(user_mapping)
    return df


def build_user_item_matrix(df_prep, log_fn=None):
    if log_fn is None: log_fn = print
    df_transaksi = (
        df_prep.groupby(['id_user','kode_produk'], as_index=False)
        .agg(quantity=('quantity','sum'), tanggal_terakhir=('tanggal_pesanan','max'))
    )
    user_filter = df_transaksi.groupby('id_user')['kode_produk'].nunique()
    # Turunkan threshold otomatis jika data kecil
    for min_item in [3, 2, 1]:
        user_valid = user_filter[user_filter >= min_item].index
        if len(user_valid) > 0:
            if min_item < 3:
                log_fn(f"    ⚠️ Filter diturunkan ke ≥{min_item} produk/user karena data terbatas")
            break
    df_transaksi = df_transaksi[df_transaksi['id_user'].isin(user_valid)]
    df_transaksi['interaction'] = np.log1p(df_transaksi['quantity'])

    train_list, test_list = [], []
    for user_id, group in df_transaksi.groupby('id_user'):
        if len(group) < 2:
            train_list.append(group)
            continue
        test_row   = group.nlargest(1, 'tanggal_terakhir')
        train_rows = group.drop(test_row.index)
        train_list.append(train_rows)
        test_list.append(test_row)

    if not train_list:
        raise ValueError("Data terlalu sedikit untuk diproses. Pastikan file Excel tidak kosong dan memiliki transaksi yang valid.")
    train_data = pd.concat(train_list, ignore_index=True)
    test_data  = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    user_item_matrix = train_data.pivot_table(
        index='id_user', columns='kode_produk', values='interaction', fill_value=0
    )
    return user_item_matrix, train_data, test_data


def build_item_features(df_prep):
    df_item = (
        df_prep[['kode_produk','nama_produk','kategori','harga','berat','platform']]
        .drop_duplicates(subset='kode_produk').reset_index(drop=True)
    )
    df_item['berat']    = df_item['berat'].fillna(df_item['berat'].median())
    df_item['harga']    = df_item['harga'].fillna(df_item['harga'].median())
    df_item['kategori'] = df_item['kategori'].fillna('Lainnya')

    tfidf = TfidfVectorizer()
    nama_produk_features = tfidf.fit_transform(df_item['nama_produk'])

    encoder = OneHotEncoder(handle_unknown='ignore')
    kategori_platform_features = encoder.fit_transform(df_item[['kategori','platform']])

    scaler = MinMaxScaler()
    numeric_features = scaler.fit_transform(df_item[['harga','berat']])

    item_features = hstack([
        nama_produk_features,
        kategori_platform_features,
        csr_matrix(numeric_features)
    ])
    return df_item, item_features, tfidf, encoder, scaler


def run_preprocessing(tiktok_path, shopee_path, log_fn=None):
    if log_fn is None:
        log_fn = print

    log_fn("📂 [1/6] Membaca file Excel...")
    dfs = []
    if tiktok_path:
        dfs.append(load_platform(tiktok_path, COLUMN_MAP_TIKTOK, "tiktok"))
        log_fn("    → TikTok Shop berhasil dibaca")
    if shopee_path:
        dfs.append(load_platform(shopee_path, COLUMN_MAP_SHOPEE, "shopee"))
        log_fn("    → Shopee berhasil dibaca")
    if not dfs:
        raise ValueError("Tidak ada file yang bisa diproses")
    df_raw = pd.concat(dfs, ignore_index=True)
    log_fn(f"    → Total baris gabungan: {len(df_raw):,}")

    log_fn("🧹 [2/6] Cleaning data...")
    df_clean = clean_data(df_raw)
    log_fn(f"    → Baris setelah cleaning: {len(df_clean):,}")

    log_fn("⚙️  [3/6] Preparation & standarisasi...")
    df_prep = prepare_data(df_clean)
    # Catatan: user ID mapping BELUM dilakukan di sini.
    # Mapping dilakukan di app.py setelah digabung dengan data lama,
    # agar username yang sama selalu dapat ID yang konsisten.
    log_fn(f"    → User unik (sebelum mapping): {df_prep['id_user'].nunique():,}")
    log_fn(f"    → Produk unik: {df_prep['kode_produk'].nunique():,}")

    # Matrix dan features TIDAK dibangun di sini.
    # Akan dibangun di app.py setelah data digabung dengan data lama
    # dan user ID mapping diterapkan dari seluruh data gabungan.
    log_fn("✅ Preprocessing data baru selesai (siap digabung)!")
    return {
        "df_prep": df_prep,
    }