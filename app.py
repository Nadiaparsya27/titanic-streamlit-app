import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Eksplorasi Dataset Titanic",
    page_icon="🚢",
    layout="wide",
)


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


df_train = load_data("train.csv")
df_test = load_data("test.csv")

st.title("🚢 Dashboard Eksplorasi Dataset Titanic")
st.caption("Implementasi Aplikasi Data Interaktif Berbasis Streamlit")
st.divider()

# Ringkasan metrik utama pada data training
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Penumpang (Train)", f"{len(df_train):,}")
m2.metric("Jumlah Selamat", f"{int(df_train['Survived'].sum()):,}")
m3.metric("Tingkat Keselamatan", f"{df_train['Survived'].mean() * 100:.1f}%")
m4.metric("Rata-rata Tarif (Fare)", f"{df_train['Fare'].mean():.2f}")

st.divider()

# Sidebar untuk filter global & info singkat
st.sidebar.header("🎛️ Filter Sidebar")
selected_sex = st.sidebar.radio(
    "Pilih Jenis Kelamin:",
    options=["Semua", "female", "male"],
    index=0,
)

# Filter interaktif di area utama
st.header("1. Filter Interaktif Penumpang")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    pclass_options = sorted(df_train["Pclass"].dropna().unique().tolist())
    pclass = st.selectbox("Pilih Kelas Penumpang (Pclass):", pclass_options)

with col_f2:
    age_min = int(np.floor(df_train["Age"].min()))
    age_max = int(np.ceil(df_train["Age"].max()))
    age_range = st.slider(
        "Pilih Rentang Usia:",
        min_value=age_min,
        max_value=age_max,
        value=(20, 40),
    )

with col_f3:
    name_query = st.text_input("Cari Nama Penumpang (opsional):")

filtered_df = df_train[
    (df_train["Pclass"] == pclass)
    & (df_train["Age"].between(age_range[0], age_range[1], inclusive="both"))
]

if selected_sex != "Semua":
    filtered_df = filtered_df[filtered_df["Sex"] == selected_sex]

if name_query.strip():
    filtered_df = filtered_df[
        filtered_df["Name"].str.contains(name_query.strip(), case=False, na=False)
    ]

st.write(f"Menampilkan **{len(filtered_df)}** penumpang sesuai filter:")
st.dataframe(filtered_df, use_container_width=True)

st.divider()

# Visualisasi dalam tab agar tampilan tidak terlalu panjang ke bawah
st.header("2. Visualisasi Data")
tab1, tab2, tab3 = st.tabs(
    ["Distribusi Keselamatan", "Tarif Berdasarkan Usia", "Distribusi Usia"]
)

with tab1:
    survived_counts = df_train["Survived"].value_counts().rename(
        index={0: "Tidak Selamat (0)", 1: "Selamat (1)"}
    )
    st.bar_chart(survived_counts, use_container_width=True)

with tab2:
    fare_by_age = df_train.groupby("Age")["Fare"].mean().dropna()
    st.line_chart(fare_by_age, use_container_width=True)

with tab3:
    age_counts = df_train["Age"].value_counts().sort_index()
    st.area_chart(age_counts, use_container_width=True)

st.divider()

# Dataset & statistik deskriptif
st.header("3. Pratinjau Dataset & Statistik Deskriptif")
col_d1, col_d2 = st.columns(2)
with col_d1:
    st.subheader("Data Training (5 Baris Awal)")
    st.dataframe(df_train.head(), use_container_width=True)
with col_d2:
    st.subheader("Data Testing (5 Baris Awal)")
    st.dataframe(df_test.head(), use_container_width=True)

with st.expander("Lihat Statistik Deskriptif Data Training"):
    st.dataframe(df_train.describe(), use_container_width=True)