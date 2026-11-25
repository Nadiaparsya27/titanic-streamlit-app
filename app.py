import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="TITANIC PASSENGER SURVIVAL",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS (MENYERUPAI TAMPILAN REFERENSI)
# ==============================================================================
st.markdown(
    """
    <style>
    /* Latar utama & sidebar */
    .stApp {
        background-color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fb;
        border-right: 1px solid #e6e9ef;
    }

    /* Hero Banner Gradien */
    .hero-banner {
        background: linear-gradient(90deg, #0b192c 0%, #104e5b 55%, #11998e 100%);
        padding: 28px 34px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 22px;
        box-shadow: 0 4px 12px rgba(11, 25, 44, 0.08);
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin: 0 0 8px 0;
        text-transform: uppercase;
    }
    .hero-banner p {
        color: #e2e8f0;
        font-size: 0.96rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Judul Sub-bagian Identitas */
    .section-label {
        font-size: 0.92rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 10px;
    }

    /* Kotak Kode Kelas */
    .class-badge {
        display: inline-block;
        background-color: #f1f5f9;
        color: #0f766e;
        font-family: 'Courier New', monospace;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 6px 14px;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        margin-bottom: 8px;
    }
    .class-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* Pill Anggota Kelompok */
    .member-pill {
        display: inline-block;
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 7px 14px;
        margin: 0 8px 8px 0;
        font-size: 0.86rem;
        color: #1e293b;
    }
    .member-pill strong {
        font-weight: 700;
        color: #0f172a;
    }
    .member-pill span {
        color: #64748b;
        font-family: 'Courier New', monospace;
        font-weight: 600;
    }

    /* Kartu KPI dengan Garis Aksen Hijau di Kiri */
    .kpi-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0d9488;
        border-radius: 10px;
        padding: 16px 20px;
        height: 100%;
    }
    .kpi-title {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }

    /* Breadcrumb Bar */
    .breadcrumb-bar {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0d9488;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.85rem;
        color: #0f766e;
        font-weight: 600;
    }

    /* Footer Sidebar */
    .sidebar-footer {
        margin-top: 24px;
        padding-top: 16px;
        border-top: 1px solid #e2e8f0;
    }
    .sidebar-footer-title {
        font-size: 0.85rem;
        font-weight: 800;
        color: #0f172a;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .sidebar-footer-sub {
        font-size: 0.78rem;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# LOAD DATA (DENGAN CACHE)
# ==============================================================================
@st.cache_data
def load_data():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    return df_train, df_test


df_train, df_test = load_data()

# ==============================================================================
# SIDEBAR NAVIGASI (4 KATEGORI / 7 HALAMAN)
# ==============================================================================
if "active_page" not in st.session_state:
    st.session_state.active_page = "1. Beranda"

menu_structure = {
    "Persiapan & Dataset": [
        "1. Beranda",
        "2. Struktur Data & Filter",
    ],
    "Eksplorasi & Diagnostik EDA": [
        "3. Statistik, Korelasi & Outlier",
        "4. Galeri Visualisasi",
        "5. Simulasi Masalah Preprocessing",
    ],
    "Pemodelan Machine Learning": [
        "6. Evaluasi Algoritma Klasifikasi",
    ],
    "Prediksi & Laporan Akhir": [
        "7. Prediksi Penumpang & Kesimpulan",
    ],
}

for category, pages in menu_structure.items():
    st.sidebar.markdown(
        f"<div style='font-size:0.8rem; font-weight:700; color:#475569; margin:12px 0 6px 0;'>{category}</div>",
        unsafe_allow_html=True,
    )
    for page in pages:
        is_selected = st.session_state.active_page == page
        btn_type = "primary" if is_selected else "secondary"
        if st.sidebar.button(
            page,
            key=f"nav_{page}",
            use_container_width=True,
            type=btn_type,
        ):
            st.session_state.active_page = page
            st.rerun()

st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <div class="sidebar-footer-title">TITANIC SURVIVAL PREDICTION</div>
        <div class="sidebar-footer-sub">Klasifikasi Keselamatan Penumpang • Kelas 05TPLP016</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# HEADER HERO BANNER + IDENTITAS KELAS & ANGGOTA (TAMPIL DI ATAS)
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <h1>TITANIC PASSENGER SURVIVAL</h1>
        <p>Eksplorasi Data (EDA), Diagnostik Masalah Preprocessing, dan Klasifikasi Machine Learning pada Dataset Penumpang Titanic.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_id1, col_id2 = st.columns([1, 2.2])

with col_id1:
    st.markdown('<div class="section-label">Identitas Kelas</div>', unsafe_allow_html=True)
    st.markdown('<div class="class-badge">05TPLP016</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="class-sub">Teknik Informatika • Mata Kuliah Machine Learning</div>',
        unsafe_allow_html=True,
    )

with col_id2:
    st.markdown(
        '<div class="section-label">Anggota Kelompok Penyusun</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div>
            <div class="member-pill"><strong>Nadia Parsya Rachmadiany</strong> • <span>241011401675</span></div>
            <div class="member-pill"><strong>Putri Adinda Siregar</strong> • <span>241011402485</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 4 KARTU KPI RINGKASAN DATASET
# ==============================================================================
total_obs = len(df_train)
total_features = df_train.shape[1] - 2  # Tanpa PassengerId & Survived
not_survived = int((df_train["Survived"] == 0).sum())
survived = int((df_train["Survived"] == 1).sum())
pct_not_survived = (not_survived / total_obs) * 100
pct_survived = (survived / total_obs) * 100

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Observasi Penumpang</div>
            <div class="kpi-value">{total_obs} sampel</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Fitur Prediktor Utama</div>
            <div class="kpi-value">{total_features} atribut</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Tidak Selamat (0)</div>
            <div class="kpi-value">{not_survived} ({pct_not_survived:.2f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Selamat (1)</div>
            <div class="kpi-value">{survived} ({pct_survived:.2f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin: 22px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

active = st.session_state.active_page

# ==============================================================================
# HALAMAN 1: BERANDA
# ==============================================================================
if active == "1. Beranda":
    sub_menu = st.radio(
        "Pilih Sub-Menu Halaman:",
        ["Ringkasan Alur", "Dataset", "Instalasi"],
        horizontal=True,
    )

    st.markdown(
        f'<div class="breadcrumb-bar">Persiapan & Dataset › Beranda › {sub_menu}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    if sub_menu == "Ringkasan Alur":
        c_left, c_right = st.columns([1.3, 1])
        with c_left:
            st.subheader("Peta Menu Aplikasi (4 Kategori / 7 Halaman)")
            st.markdown(
                """
                - **Persiapan & Dataset:** Ringkasan proyek, struktur kolom `train.csv` & `test.csv`, serta filter interaktif penumpang.
                - **Eksplorasi & Diagnostik EDA:** Statistik deskriptif, matriks korelasi, analisis *outlier*, visualisasi distribusi, dan deteksi *missing values*.
                - **Pemodelan Machine Learning:** Pelatihan dan perbandingan performa algoritma (*Logistic Regression*, *Decision Tree*, *Random Forest*).
                - **Prediksi & Laporan Akhir:** Simulasi prediksi keselamatan penumpang baru secara *real-time* dan kesimpulan akhir.
                """
            )
        with c_right:
            st.subheader("Formula Matematis Utama")
            st.markdown("**1. Standardisasi Fitur Numerik (Z-Score Scaling):**")
            st.latex(r"z = \frac{x - \mu}{\sigma}")
            st.markdown("**2. Probabilitas Logistic Regression (Sigmoid):**")
            st.latex(r"P(Y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^{n} \beta_i X_i)}}")

    elif sub_menu == "Dataset":
        st.subheader("Pratinjau Dataset Titanic (`train.csv` & `test.csv`)")
        tab_tr, tab_te = st.tabs(["Data Training (891 baris)", "Data Testing (418 baris)"])
        with tab_tr:
            st.dataframe(df_train.head(20), use_container_width=True)
        with tab_te:
            st.dataframe(df_test.head(20), use_container_width=True)

    elif sub_menu == "Instalasi":
        st.subheader("Panduan Menjalankan Aplikasi Secara Lokal")
        st.code(
            """pip install streamlit pandas numpy matplotlib scikit-learn
streamlit run app.py""",
            language="bash",
        )

# ==============================================================================
# HALAMAN 2: STRUKTUR DATA & FILTER
# ==============================================================================
elif active == "2. Struktur Data & Filter":
    st.markdown(
        '<div class="breadcrumb-bar">Persiapan & Dataset › 2. Struktur Data & Filter</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Filter Interaktif Data Penumpang")

    f1, f2, f3 = st.columns(3)
    with f1:
        pclass = st.selectbox(
            "Kelas Tiket (Pclass):",
            options=["Semua", 1, 2, 3],
        )
    with f2:
        sex = st.selectbox(
            "Jenis Kelamin (Sex):",
            options=["Semua", "female", "male"],
        )
    with f3:
        age_range = st.slider(
            "Rentang Usia (Age):",
            min_value=0,
            max_value=80,
            value=(0, 80),
        )

    filtered = df_train.copy()
    if pclass != "Semua":
        filtered = filtered[filtered["Pclass"] == pclass]
    if sex != "Semua":
        filtered = filtered[filtered["Sex"] == sex]
    filtered = filtered[
        filtered["Age"].fillna(-1).between(age_range[0], age_range[1])
        | ((age_range == (0, 80)) & filtered["Age"].isna())
    ]

    st.write(f"Menampilkan **{len(filtered)}** data penumpang sesuai kriteria:")
    st.dataframe(filtered, use_container_width=True)

# ==============================================================================
# HALAMAN 3: STATISTIK, KORELASI & OUTLIER
# ==============================================================================
elif active == "3. Statistik, Korelasi & Outlier":
    st.markdown(
        '<div class="breadcrumb-bar">Eksplorasi & Diagnostik EDA › 3. Statistik, Korelasi & Outlier</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Statistik Deskriptif Fitur Numerik")
    st.dataframe(df_train.describe().T, use_container_width=True)

    st.subheader("Matriks Korelasi Fitur Numerik")
    num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    corr = df_train[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(corr, cmap="Teal", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=30, ha="right")
    ax.set_yticklabels(num_cols)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

# ==============================================================================
# HALAMAN 4: GALERI VISUALISASI
# ==============================================================================
elif active == "4. Galeri Visualisasi":
    st.markdown(
        '<div class="breadcrumb-bar">Eksplorasi & Diagnostik EDA › 4. Galeri Visualisasi</div>',
        unsafe_allow_html=True,
    )
    v1, v2 = st.columns(2)
    with v1:
        st.subheader("Tingkat Keselamatan Berdasarkan Jenis Kelamin")
        surv_sex = df_train.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)
        st.bar_chart(surv_sex, use_container_width=True)
    with v2:
        st.subheader("Tingkat Keselamatan Berdasarkan Kelas Tiket")
        surv_class = df_train.groupby(["Pclass", "Survived"]).size().unstack(fill_value=0)
        st.bar_chart(surv_class, use_container_width=True)

# ==============================================================================
# HALAMAN 5: SIMULASI MASALAH PREPROCESSING
# ==============================================================================
elif active == "5. Simulasi Masalah Preprocessing":
    st.markdown(
        '<div class="breadcrumb-bar">Eksplorasi & Diagnostik EDA › 5. Simulasi Masalah Preprocessing</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Diagnostik Missing Values pada Dataset Training")
    missing_df = pd.DataFrame(
        {
            "Kolom": df_train.columns,
            "Jumlah Missing": df_train.isna().sum().values,
            "Persentase (%)": (df_train.isna().mean().values * 100).round(2),
        }
    )
    st.dataframe(missing_df, use_container_width=True)
    st.info(
        "Strategi Preprocessing: Kolom `Age` diimputasi menggunakan median, `Embarked` menggunakan modus, `Cabin` dieliminasi karena >77% kosong, dan fitur kategorikal di-encode dengan One-Hot Encoding."
    )

# ==============================================================================
# HALAMAN 6: EVALUASI ALGORITMA KLASIFIKASI
# ==============================================================================
elif active == "6. Evaluasi Algoritma Klasifikasi":
    st.markdown(
        '<div class="breadcrumb-bar">Pemodelan Machine Learning › 6. Evaluasi Algoritma Klasifikasi</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Perbandingan Performa Model Machine Learning")

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df_train[features]
    y = df_train["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    cat_features = ["Sex", "Embarked"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_features,
            ),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
                cat_features,
            ),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
    }

    results = []
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        acc = accuracy_score(y_val, preds)
        results.append({"Algoritma": name, "Akurasi Validasi (%)": round(acc * 100, 2)})

    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True)
    st.bar_chart(res_df.set_index("Algoritma"), use_container_width=True)

# ==============================================================================
# HALAMAN 7: PREDIKSI PENUMPANG & KESIMPULAN
# ==============================================================================
elif active == "7. Prediksi Penumpang & Kesimpulan":
    st.markdown(
        '<div class="breadcrumb-bar">Prediksi & Laporan Akhir › 7. Prediksi Penumpang & Kesimpulan</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Simulasi Prediksi Keselamatan Penumpang")

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df_train[features]
    y = df_train["Survived"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                ["Pclass", "Age", "SibSp", "Parch", "Fare"],
            ),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
                ["Sex", "Embarked"],
            ),
        ]
    )

    rf_pipeline = Pipeline(
        [("prep", preprocessor), ("model", RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42))]
    )
    rf_pipeline.fit(X, y)

    with st.form("form_prediksi"):
        p1, p2, p3 = st.columns(3)
        with p1:
            in_pclass = st.selectbox("Kelas Tiket (Pclass)", [1, 2, 3], index=0)
            in_sex = st.selectbox("Jenis Kelamin", ["female", "male"], index=0)
        with p2:
            in_age = st.number_input("Usia (Tahun)", min_value=0.0, max_value=80.0, value=25.0)
            in_fare = st.number_input("Tarif Tiket (Fare)", min_value=0.0, max_value=520.0, value=50.0)
        with p3:
            in_sibsp = st.number_input("Jumlah Saudara/Pasangan (SibSp)", min_value=0, max_value=8, value=0)
            in_parch = st.number_input("Jumlah Orang Tua/Anak (Parch)", min_value=0, max_value=6, value=0)
            in_embarked = st.selectbox("Pelabuhan Keberangkatan", ["S", "C", "Q"], index=0)

        submitted = st.form_submit_button("Prediksi Sekarang")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "Pclass": in_pclass,
                    "Sex": in_sex,
                    "Age": in_age,
                    "SibSp": in_sibsp,
                    "Parch": in_parch,
                    "Fare": in_fare,
                    "Embarked": in_embarked,
                }
            ]
        )
        pred = rf_pipeline.predict(input_df)[0]
        proba = rf_pipeline.predict_proba(input_df)[0]

        if pred == 1:
            st.success(f"Hasil Prediksi: **SELAMAT (1)** — Probabilitas Selamat: **{proba[1]*100:.2f}%**")
        else:
            st.error(f"Hasil Prediksi: **TIDAK SELAMAT (0)** — Probabilitas Selamat: **{proba[1]*100:.2f}%**")