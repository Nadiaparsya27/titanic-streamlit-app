import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="TITANIC ML • Blocs Store Style",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# CUSTOM CSS (DARK MODE + CORAL-VIOLET GRADIENT HERO SEPERTI REFERENSI)
# ==============================================================================
st.markdown(
    """
    <style>
    /* Sembunyikan sidebar & header bawaan agar tampilan full-bleed seperti web modern */
    section[data-testid="stSidebar"] {
        display: none;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        max-width: 1240px;
    }

    /* Latar Belakang Utama Hitam Pekat */
    .stApp {
        background-color: #050508;
        color: #f8fafc;
    }

    /* Hero Banner Gradien Coral-Pink ke Electric-Violet */
    .hero-wrapper {
        position: relative;
        background: linear-gradient(115deg, #ff766c 0%, #d442b8 48%, #4824df 100%);
        border-radius: 18px;
        padding: 26px 40px 46px 40px;
        margin-bottom: 26px;
        overflow: hidden;
        box-shadow: 0 14px 40px rgba(72, 36, 223, 0.28);
    }

    /* Dekorasi Lingkaran & Garis Geometris Halus seperti di Gambar */
    .hero-circle-1 {
        position: absolute;
        width: 340px;
        height: 340px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
        top: -60px;
        left: 34%;
        pointer-events: none;
    }
    .hero-circle-2 {
        position: absolute;
        width: 190px;
        height: 190px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.06);
        bottom: -70px;
        right: 8%;
        pointer-events: none;
    }
    .hero-line-1 {
        position: absolute;
        width: 240px;
        height: 1px;
        background: rgba(255, 255, 255, 0.28);
        transform: rotate(-45deg);
        bottom: 45px;
        left: 11%;
        pointer-events: none;
    }
    .hero-line-2 {
        position: absolute;
        width: 260px;
        height: 1px;
        background: rgba(255, 255, 255, 0.28);
        transform: rotate(-45deg);
        bottom: 65px;
        right: 12%;
        pointer-events: none;
    }

    /* Top Navbar di dalam Hero */
    .hero-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 34px;
        position: relative;
        z-index: 2;
        flex-wrap: wrap;
        gap: 12px;
    }
    .brand-logo {
        font-size: 1.15rem;
        font-weight: 900;
        letter-spacing: 2px;
        color: #ffffff;
        text-transform: uppercase;
    }
    .brand-logo span {
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 1px;
        opacity: 0.85;
        margin-left: 4px;
    }
    .top-nav-right {
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 0.84rem;
        color: #ffffff;
    }
    .nav-pill-btn {
        background: rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(6px);
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }

    /* Teks Tengah Hero */
    .hero-center {
        text-align: center;
        position: relative;
        z-index: 2;
        max-width: 780px;
        margin: 0 auto 22px auto;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 10px;
        line-height: 1.25;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: rgba(255, 255, 255, 0.92);
        font-weight: 400;
        margin-bottom: 0;
    }

    /* Identitas Kelompok di Bawah Subtitle Hero */
    .member-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
        position: relative;
        z-index: 2;
    }
    .member-chip {
        background: rgba(15, 15, 25, 0.34);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.84rem;
    }
    .member-chip strong {
        font-weight: 700;
    }
    .member-chip code {
        background: rgba(255, 255, 255, 0.16);
        color: #fff;
        padding: 2px 7px;
        border-radius: 6px;
        margin-left: 5px;
        font-size: 0.8rem;
    }

    /* Styling Search Bar Streamlit agar menjadi Pill Putih di Tengah */
    div[data-testid="stTextInput"] {
        max-width: 660px;
        margin: -54px auto 34px auto;
        position: relative;
        z-index: 10;
    }
    div[data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border-radius: 999px !important;
        padding: 14px 24px !important;
        font-size: 0.95rem !important;
        border: none !important;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35) !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #94a3b8 !important;
    }
    div[data-testid="stTextInput"] label {
        display: none !important;
    }

    /* Navigasi Tab Horizontal ala "Featured | New releases | Blocs | ..." */
    div[data-testid="stRadio"] > div {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 8px;
        background: transparent;
        margin-bottom: 24px;
    }
    div[data-testid="stRadio"] label {
        background: transparent !important;
        border: none !important;
        padding: 8px 14px !important;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label p {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"] p {
        color: #ffffff !important;
        border-bottom: 2px solid #7c5cff;
        padding-bottom: 6px;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* 4 Kartu Showcase Gradien (Menyerupai 4 Kartu Bawah di Gambar) */
    .showcase-card {
        border-radius: 12px;
        padding: 28px 20px;
        text-align: center;
        color: #ffffff;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease;
    }
    .showcase-card:hover {
        transform: translateY(-4px);
    }
    .card-purple-1 {
        background: linear-gradient(160deg, #5b189a 0%, #240046 100%);
        border: 1px solid rgba(168, 85, 247, 0.35);
    }
    .card-purple-2 {
        background: linear-gradient(160deg, #3c096c 0%, #10002b 100%);
        border: 1px solid rgba(139, 92, 246, 0.35);
    }
    .card-peach {
        background: linear-gradient(180deg, #9d8df1 0%, #f7b29c 100%);
        color: #1e1b4b;
    }
    .card-cyan {
        background: linear-gradient(160deg, #00f5d4 0%, #0066ff 100%);
        color: #ffffff;
    }
    .hex-badge {
        width: 64px;
        height: 64px;
        border-radius: 16px;
        background: rgba(15, 15, 25, 0.72);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 16px;
        border: 2px solid rgba(255, 255, 255, 0.25);
        color: #ffffff;
    }
    .card-big-num {
        font-size: 1.65rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .card-caption {
        font-size: 0.85rem;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        opacity: 0.9;
        font-weight: 600;
    }

    /* Container Konten Gelap */
    .dark-panel {
        background-color: #0e0e14;
        border: 1px solid #1e1e2d;
        border-radius: 14px;
        padding: 22px 26px;
        margin-top: 12px;
    }
    h2, h3, h4 {
        color: #ffffff !important;
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
# HERO SECTION (CORAL-PINK TO VIOLET GRADIENT + IDENTITAS)
# ==============================================================================
st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-circle-1"></div>
        <div class="hero-circle-2"></div>
        <div class="hero-line-1"></div>
        <div class="hero-line-2"></div>
        <div class="hero-topbar">
            <div class="brand-logo">TITANIC<span>ML STUDIO</span></div>
            <div class="top-nav-right">
                <span>Mata Kuliah: <strong>Machine Learning</strong></span>
                <span class="nav-pill-btn">Kelas 05TPLP016</span>
            </div>
        </div>
        <div class="hero-center">
            <div class="hero-title">Energize your Titanic analysis with Machine Learning.</div>
            <p class="hero-subtitle">Eksplorasi Data (EDA), Diagnostik Preprocessing, dan Prediksi Keselamatan Penumpang Titanic.</p>
            <div class="member-row">
                <div class="member-chip"><strong>Nadia Parsya Rachmadiany</strong><code>241011401675</code></div>
                <div class="member-chip"><strong>Putri Adinda Siregar</strong><code>241011402485</code></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search Bar Pill Putih di Tengah (Menyerupai "Search for Blocs, Brics, Templates and more...")
quick_search = st.text_input(
    "Cari Penumpang",
    placeholder="🔍  Cari nama penumpang Titanic (contoh: Allen, Braund, Morrell)...",
)

if quick_search.strip():
    search_res = df_train[
        df_train["Name"].str.contains(quick_search.strip(), case=False, na=False)
    ]
    st.info(f"Hasil pencarian cepat untuk **'{quick_search}'**: ditemukan **{len(search_res)}** penumpang.")
    st.dataframe(
        search_res[["PassengerId", "Name", "Sex", "Age", "Pclass", "Fare", "Survived"]],
        use_container_width=True,
    )

# ==============================================================================
# NAVIGASI HORIZONTAL MINIMALIS (ALA FEATURED | NEW RELEASES | BLOCS | ...)
# ==============================================================================
menu_options = [
    "Featured (Beranda)",
    "Dataset & Filter",
    "Statistik & Korelasi",
    "Galeri Visualisasi",
    "Preprocessing EDA",
    "Evaluasi Model",
    "Prediksi Penumpang",
]

selected_tab = st.radio(
    "Navigasi Utama",
    menu_options,
    horizontal=True,
    label_visibility="collapsed",
)

# ==============================================================================
# KONTEN TIAP MENU
# ==============================================================================
if selected_tab == "Featured (Beranda)":
    total_obs = len(df_train)
    total_feat = df_train.shape[1] - 2
    not_surv = int((df_train["Survived"] == 0).sum())
    surv = int((df_train["Survived"] == 1).sum())

    # 4 Kartu Gradien di Bagian Bawah (Menyerupai BLOCSMASTER, BLOCSTEMPLATES, dll.)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="showcase-card card-purple-1">
                <div class="hex-badge">🚢</div>
                <div class="card-big-num">{total_obs} Sampel</div>
                <div class="card-caption">TOTAL PENUMPANG</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="showcase-card card-purple-2">
                <div class="hex-badge">🧬</div>
                <div class="card-big-num">{total_feat} Atribut</div>
                <div class="card-caption">FITUR DATASET</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="showcase-card card-peach">
                <div class="hex-badge">⚠️</div>
                <div class="card-big-num">{not_surv} ({not_surv/total_obs*100:.1f}%)</div>
                <div class="card-caption">TIDAK SELAMAT (0)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="showcase-card card-cyan">
                <div class="hex-badge">✨</div>
                <div class="card-big-num">{surv} ({surv/total_obs*100:.1f}%)</div>
                <div class="card-caption">SELAMAT (1)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    col_info1, col_info2 = st.columns([1.2, 1])
    with col_info1:
        st.subheader("Ringkasan Alur Analisis & Pemodelan")
        st.markdown(
            """
            - **Dataset & Filter:** Eksplorasi `train.csv` (891 penumpang) & `test.csv` (418 penumpang) berdasarkan kelas tiket, gender, dan usia.
            - **Statistik & Korelasi:** Analisis distribusi numerik dan matriks korelasi antar fitur.
            - **Preprocessing EDA:** Penanganan *missing values* (`Age`, `Embarked`, `Cabin`) serta *One-Hot Encoding*.
            - **Evaluasi & Prediksi:** Perbandingan akurasi *Logistic Regression*, *Decision Tree*, dan *Random Forest* beserta simulasi prediksi penumpang.
            """
        )
    with col_info2:
        st.subheader("Formula Matematis Utama")
        st.markdown("**Standardisasi Z-Score:**")
        st.latex(r"z = \frac{x - \mu}{\sigma}")
        st.markdown("**Probabilitas Sigmoid (Logistic Regression):**")
        st.latex(r"P(Y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^{n} \beta_i X_i)}}")

elif selected_tab == "Dataset & Filter":
    st.subheader("Filter Interaktif Data Penumpang")
    f1, f2, f3 = st.columns(3)
    with f1:
        pclass = st.selectbox("Kelas Tiket (Pclass):", ["Semua", 1, 2, 3])
    with f2:
        sex = st.selectbox("Jenis Kelamin (Sex):", ["Semua", "female", "male"])
    with f3:
        age_range = st.slider("Rentang Usia (Age):", 0, 80, (0, 80))

    filtered = df_train.copy()
    if pclass != "Semua":
        filtered = filtered[filtered["Pclass"] == pclass]
    if sex != "Semua":
        filtered = filtered[filtered["Sex"] == sex]
    filtered = filtered[
        filtered["Age"].fillna(-1).between(age_range[0], age_range[1])
        | ((age_range == (0, 80)) & filtered["Age"].isna())
    ]

    st.write(f"Menampilkan **{len(filtered)}** penumpang:")
    st.dataframe(filtered, use_container_width=True)

elif selected_tab == "Statistik & Korelasi":
    st.subheader("Statistik Deskriptif Fitur Numerik")
    st.dataframe(df_train.describe().T, use_container_width=True)

    st.subheader("Matriks Korelasi")
    num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    corr = df_train[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0e0e14")
    ax.set_facecolor("#0e0e14")
    im = ax.imshow(corr, cmap="cool", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=30, ha="right", color="white")
    ax.set_yticklabels(num_cols, color="white")
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    st.pyplot(fig)

elif selected_tab == "Galeri Visualisasi":
    v1, v2 = st.columns(2)
    with v1:
        st.subheader("Keselamatan Berdasarkan Jenis Kelamin")
        surv_sex = df_train.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)
        st.bar_chart(surv_sex, use_container_width=True)
    with v2:
        st.subheader("Keselamatan Berdasarkan Kelas Tiket")
        surv_class = df_train.groupby(["Pclass", "Survived"]).size().unstack(fill_value=0)
        st.bar_chart(surv_class, use_container_width=True)

elif selected_tab == "Preprocessing EDA":
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
        "Strategi Preprocessing: `Age` diimputasi menggunakan median, `Embarked` menggunakan modus, `Cabin` dieliminasi (>77% kosong), dan variabel kategorikal di-encode memakai One-Hot Encoding."
    )

elif selected_tab == "Evaluasi Model":
    st.subheader("Perbandingan Akurasi Model Klasifikasi")
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df_train[features]
    y = df_train["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
    }

    results = []
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_val, pipe.predict(X_val))
        results.append({"Algoritma": name, "Akurasi Validasi (%)": round(acc * 100, 2)})

    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True)
    st.bar_chart(res_df.set_index("Algoritma"), use_container_width=True)

elif selected_tab == "Prediksi Penumpang":
    st.subheader("Simulasi Prediksi Keselamatan Penumpang Baru")
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

    with st.form("form_prediksi_dark"):
        p1, p2, p3 = st.columns(3)
        with p1:
            in_pclass = st.selectbox("Kelas Tiket (Pclass)", [1, 2, 3], index=0)
            in_sex = st.selectbox("Jenis Kelamin", ["female", "male"], index=0)
        with p2:
            in_age = st.number_input("Usia (Tahun)", min_value=0.0, max_value=80.0, value=25.0)
            in_fare = st.number_input("Tarif Tiket (Fare)", min_value=0.0, max_value=520.0, value=50.0)
        with p3:
            in_sibsp = st.number_input("Saudara / Pasangan (SibSp)", min_value=0, max_value=8, value=0)
            in_parch = st.number_input("Orang Tua / Anak (Parch)", min_value=0, max_value=6, value=0)
            in_embarked = st.selectbox("Pelabuhan (Embarked)", ["S", "C", "Q"], index=0)

        submitted = st.form_submit_button("Prediksi Keselamatan")

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
            st.success(f"Hasil Prediksi: **SELAMAT (1)** — Probabilitas: **{proba[1]*100:.2f}%**")
        else:
            st.error(f"Hasil Prediksi: **TIDAK SELAMAT (0)** — Probabilitas Selamat: **{proba[1]*100:.2f}%**")