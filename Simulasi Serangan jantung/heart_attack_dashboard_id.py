"""
Dashboard: Analisis & Prediksi Risiko Serangan Jantung — Indonesia
Sumber data: heart_attack_cleaned.csv — hasil ekspor yang SUDAH DIBERSIHKAN
dari Orange (File -> Impute -> Save Data). Dashboard ini TIDAK membersihkan
data sendiri; pembersihan (mengisi nilai kosong, dll) hanya terjadi sekali,
di hulu, di Orange. Kalau file itu tidak ditemukan, aplikasi berhenti
dengan instruksi, bukan diam-diam membersihkan CSV mentah sendiri.

Jalankan: streamlit run heart_attack_dashboard_id.py

Catatan soal filter: Streamlit menjalankan ulang seluruh script setiap ada
interaksi, dan st.tabs() tidak memberi tahu kode Python tab mana yang
sedang aktif — jadi st.sidebar.slider() biasa yang diletakkan "di dalam"
sebuah tab tetap muncul di sidebar apa pun tab yang sedang dipilih secara
visual. Supaya filter di sidebar HANYA muncul di halaman Overview (dan
hilang di halaman Prediksi), pemilihan halaman di bawah memakai st.radio(),
yang nilainya BISA dibaca di kode, bukan st.tabs().
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

st.set_page_config(page_title="Dashboard Serangan Jantung — Indonesia", layout="wide")

DEFAULT_FILE = "heart_attack_cleaned.csv"  # hasil ekspor Orange: File -> Impute -> Save Data
MODEL_FILE = "heart_attack_model.joblib"


NUMERIC_COLS = [
    "age", "cholesterol_level", "waist_circumference", "sleep_hours",
    "blood_pressure_systolic", "blood_pressure_diastolic", "fasting_blood_sugar",
    "cholesterol_hdl", "cholesterol_ldl", "triglycerides",
]
BINARY_COLS = [
    "hypertension", "diabetes", "obesity", "family_history",
    "previous_heart_disease", "medication_usage", "participated_in_free_screening",
    "heart_attack",
]


@st.cache_data
def load_data(file):
    """Memuat data yang diasumsikan sudah bersih (dari Orange). Satu-satunya
    transformasi di sini adalah pengelompokan age_group — detail khusus
    tampilan untuk chart Overview, bukan langkah pembersihan data, jadi
    tetap lokal di sisi tool yang menampilkan data."""
    # keep_default_na=False: tanpa ini, pandas diam-diam menganggap teks
    # "None" (kategori sah untuk alcohol_consumption) sebagai data kosong —
    # membatalkan pembersihan Orange setiap kali file dimuat ulang.
    # na_values=[""] tetap menandai sel yang BENAR-BENAR kosong sebagai NaN.
    df = pd.read_csv(file, keep_default_na=False, na_values=[""], low_memory=False)

    # Widget Save Data di Orange kadang menulis 2 baris ekstra tepat
    # setelah header: baris deskriptor tipe/domain (mis. "continuous",
    # "0 1", "Female Male") dan baris role kosong -- ini metadata format
    # Orange, bukan data pasien. Deteksi dengan mencari baris pertama di
    # mana "age" benar-benar berupa angka, lalu buang semua sebelum itu.
    age_numeric = pd.to_numeric(df["age"], errors="coerce")
    first_valid = age_numeric.first_valid_index()
    if first_valid is not None and first_valid > 0:
        df = df.iloc[first_valid:].reset_index(drop=True)

    # Jaga-jaga: paksa kolom numerik benar-benar jadi angka. Menangkap
    # nilai non-numerik yang mungkin masih nyelip di file.
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_rows = df[NUMERIC_COLS + BINARY_COLS].isna().any(axis=1).sum()
    if bad_rows:
        st.sidebar.warning(
            f"{bad_rows} baris punya nilai non-numerik di kolom numerik "
            f"(kemungkinan sisa metadata dari ekspor CSV) dan dianggap kosong."
        )

    bins = [0, 30, 40, 50, 60, 70, 100]
    labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    return df


@st.cache_resource
def load_model():
    """Memuat model yang sudah dilatih. Kalau belum ada, latih sekali di
    sini sebagai fallback supaya dashboard tetap jalan walau
    train_model.py belum dijalankan manual."""
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    import subprocess
    subprocess.run(["python", "train_model.py"], check=True)
    return joblib.load(MODEL_FILE)


st.title("Dashboard Analisis & Prediksi Risiko Serangan Jantung — Indonesia")

# ---------------------------------------------------------------------------
# Sumber data — selalu memakai file default, tidak ada opsi upload.
# ---------------------------------------------------------------------------
if not os.path.exists(DEFAULT_FILE):
    st.error(
        f"**{DEFAULT_FILE}** tidak ditemukan.\n\n"
        f"Dashboard ini tidak membersihkan data sendiri — pembersihan hanya "
        f"terjadi sekali, di hulu, di Orange. Untuk memperbaiki:\n"
        f"1. Di Orange, sambungkan **File → Impute → Save Data**\n"
        f"2. Simpan hasilnya sebagai `{DEFAULT_FILE}` di folder yang sama\n"
        f"3. Muat ulang halaman ini"
    )
    st.stop()
df = load_data(DEFAULT_FILE)
st.sidebar.caption(f"Sumber data: {DEFAULT_FILE} ({len(df):,} baris)")

st.caption(
    "Dataset: 158.355 individu, 28 variabel demografi/klinis/gaya hidup. "
    "Ini adalah dataset publik bergaya Kaggle yang bersifat **sintetis**, "
    "dibuat untuk latihan pemodelan prediksi — bukan data rekam medis riil "
    "dari kementerian kesehatan atau rumah sakit. Perlakukan hasilnya "
    "sebagai eksplorasi pola data dan demo model, bukan alat diagnosis "
    "medis sungguhan."
)

# ---------------------------------------------------------------------------
# Pemilihan halaman — st.radio() dipakai (bukan st.tabs()) karena nilainya
# bisa dibaca di kode. Ini yang membuat filter sidebar di bawah HANYA
# muncul saat "Overview & Analisis" dipilih.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    div[role="radiogroup"] { gap: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
page = st.radio(
    "Tampilan",
    ["📊 Overview & Analisis", "🩺 Prediksi Individu"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

# ===========================================================================
# HALAMAN 1: OVERVIEW & ANALISIS (filter ada di sidebar, hanya di halaman ini)
# ===========================================================================
if page == "📊 Overview & Analisis":

    st.sidebar.header("Filter")
    age_range = st.sidebar.slider("Rentang usia", int(df["age"].min()), int(df["age"].max()),
                                   (int(df["age"].min()), int(df["age"].max())))
    gender_pick = st.sidebar.multiselect("Gender", sorted(df["gender"].unique()))
    region_pick = st.sidebar.multiselect("Wilayah", sorted(df["region"].unique()))
    income_pick = st.sidebar.multiselect("Tingkat pendapatan", sorted(df["income_level"].unique()))
    smoking_pick = st.sidebar.multiselect("Status merokok", sorted(df["smoking_status"].unique()))

    df_f = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]
    if gender_pick:
        df_f = df_f[df_f["gender"].isin(gender_pick)]
    if region_pick:
        df_f = df_f[df_f["region"].isin(region_pick)]
    if income_pick:
        df_f = df_f[df_f["income_level"].isin(income_pick)]
    if smoking_pick:
        df_f = df_f[df_f["smoking_status"].isin(smoking_pick)]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah individu", f"{len(df_f):,}")
    col2.metric("Prevalensi serangan jantung", f"{df_f['heart_attack'].mean()*100:.1f}%")
    col3.metric("Rata-rata usia", f"{df_f['age'].mean():.0f} tahun")
    col4.metric("Rata-rata tekanan darah sistolik", f"{df_f['blood_pressure_systolic'].mean():.0f} mmHg")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Prevalensi Serangan Jantung per Kelompok Usia")
        by_age = df_f.groupby("age_group", observed=True)["heart_attack"].mean().mul(100).reset_index()
        fig = px.bar(by_age, x="age_group", y="heart_attack",
                     labels={"heart_attack": "Prevalensi (%)", "age_group": "Kelompok usia"},
                     text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Prevalensi Serangan Jantung: Gender & Wilayah")
        by_gr = df_f.groupby(["gender", "region"])["heart_attack"].mean().mul(100).reset_index()
        fig = px.bar(by_gr, x="gender", y="heart_attack", color="region", barmode="group",
                     labels={"heart_attack": "Prevalensi (%)", "gender": "Gender"}, text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Faktor Komorbid/Riwayat vs Prevalensi Serangan Jantung")
    comorbid_cols = ["hypertension", "diabetes", "obesity", "previous_heart_disease", "family_history"]
    label_map = {
        "hypertension": "Hipertensi", "diabetes": "Diabetes", "obesity": "Obesitas",
        "previous_heart_disease": "Riwayat Sakit Jantung", "family_history": "Riwayat Keluarga",
    }
    rows = []
    for c in comorbid_cols:
        for val in [0, 1]:
            subset = df_f[df_f[c] == val]
            if len(subset):
                rows.append({"faktor": label_map[c], "status": "Ya" if val else "Tidak",
                             "prevalensi": subset["heart_attack"].mean() * 100})
    comorbid_df = pd.DataFrame(rows)
    fig = px.bar(comorbid_df, x="faktor", y="prevalensi", color="status", barmode="group",
                 labels={"prevalensi": "Prevalensi (%)", "faktor": ""}, text_auto=".1f")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Faktor Gaya Hidup vs Prevalensi Serangan Jantung")
    c3, c4 = st.columns(2)
    with c3:
        by_smoke = df_f.groupby("smoking_status")["heart_attack"].mean().mul(100).reset_index()
        fig = px.bar(by_smoke, x="smoking_status", y="heart_attack",
                     labels={"heart_attack": "Prevalensi (%)", "smoking_status": "Status merokok"},
                     text_auto=".1f", title="Status Merokok")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        by_diet = df_f.groupby(["dietary_habits", "physical_activity"])["heart_attack"].mean().mul(100).reset_index()
        fig = px.bar(by_diet, x="physical_activity", y="heart_attack", color="dietary_habits",
                     barmode="group", labels={"heart_attack": "Prevalensi (%)", "physical_activity": "Aktivitas fisik"},
                     text_auto=".1f", title="Aktivitas Fisik x Pola Makan")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Korelasi Fitur Numerik dengan Serangan Jantung")
    numeric_cols = df_f.select_dtypes(include="number").columns.drop("heart_attack")
    corr = df_f[numeric_cols].corrwith(df_f["heart_attack"]).sort_values()
    fig = px.bar(corr, orientation="h", labels={"value": "Korelasi Pearson", "index": ""})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Riwayat sakit jantung, hipertensi, diabetes, dan obesitas menunjukkan "
        "korelasi terkuat. Faktor gaya hidup (rokok di luar status 'Current', "
        "alkohol, stres, tidur) dan sebagian besar nilai lab menunjukkan korelasi "
        "mendekati nol pada dataset ini — kemungkinan karena sifat sintetis data."
    )

    st.divider()
    st.subheader("Data Mentah (setelah difilter)")
    st.dataframe(df_f.drop(columns=["age_group"]), use_container_width=True, hide_index=True)

# ===========================================================================
# HALAMAN 2: PREDIKSI INDIVIDU (tanpa filter sidebar di halaman ini)
# ===========================================================================
else:
    st.subheader("Simulator Prediksi Risiko Serangan Jantung")
    st.caption(
        "Isi profil di bawah, lalu model Machine Learning (Gradient Boosting, "
        "dilatih dari 158.355 data historis) memperkirakan probabilitas "
        "serangan jantung. Ini simulasi edukatif dari data sintetis, **bukan** alat "
        "diagnosis — hasilnya tidak menggantikan pemeriksaan medis."
    )

    model_bundle = load_model()
    pipe = model_bundle["pipeline"]
    auc = model_bundle["auc"]
    prevalence = model_bundle["prevalence"]

    with st.expander("ℹ️ Performa & metodologi model (untuk presentasi)"):
        st.markdown(
            f"""
- **Algoritma**: {model_bundle['model_name'].replace('_', ' ').title()}
- **ROC-AUC di data uji (20% hold-out)**: **{auc:.3f}** — makin dekat ke 1.0
  makin baik model membedakan kasus positif vs negatif (0.5 = tebak acak).
- **Prevalensi di data latih**: {prevalence*100:.1f}%
- **Fitur input**: {len(model_bundle['features'])} variabel (numerik, biner,
  kategorikal) — kolom yang sama seperti di halaman Overview.
- Model dilatih dengan `class_weight="balanced"` untuk menghindari bias ke
  kelas mayoritas, memakai `train_test_split` 80/20 berstratifikasi supaya
  evaluasi dilakukan pada data yang tidak pernah dilihat model saat training.
            """
        )
        imp = pd.DataFrame(model_bundle["top_features"])
        fig_imp = px.bar(
            imp.sort_values("skor"), x="skor", y="fitur", orientation="h",
            labels={"skor": "Pengaruh (feature importance)", "fitur": ""},
            title="15 Fitur Paling Berpengaruh",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    st.markdown("### Profil Pasien Simulasi")

    colA, colB, colC = st.columns(3)
    with colA:
        age = st.slider("Usia", 18, 90, 50)
        gender = st.selectbox("Gender", sorted(df["gender"].unique()))
        region = st.selectbox("Wilayah", sorted(df["region"].unique()))
        income_level = st.selectbox("Tingkat pendapatan", sorted(df["income_level"].unique()))
        smoking_status = st.selectbox("Status merokok", sorted(df["smoking_status"].unique()))
    with colB:
        hypertension = st.checkbox("Hipertensi")
        diabetes = st.checkbox("Diabetes")
        obesity = st.checkbox("Obesitas")
        family_history = st.checkbox("Riwayat keluarga sakit jantung")
        previous_heart_disease = st.checkbox("Riwayat sakit jantung sebelumnya")
        medication_usage = st.checkbox("Sedang mengonsumsi obat rutin")
        participated_in_free_screening = st.checkbox("Pernah mengikuti skrining gratis")
    with colC:
        physical_activity = st.selectbox("Aktivitas fisik", sorted(df["physical_activity"].unique()))
        dietary_habits = st.selectbox("Pola makan", sorted(df["dietary_habits"].unique()))
        alcohol_consumption = st.selectbox("Konsumsi alkohol", sorted(df["alcohol_consumption"].unique()))
        air_pollution_exposure = st.selectbox("Paparan polusi udara", sorted(df["air_pollution_exposure"].unique()))
        stress_level = st.selectbox("Tingkat stres", sorted(df["stress_level"].unique()))
        ekg_results = st.selectbox("Hasil EKG", sorted(df["EKG_results"].unique()))

    st.markdown("### Nilai Klinis")
    colD, colE, colF = st.columns(3)
    with colD:
        cholesterol_level = st.slider("Kolesterol total", 100, 350, 200)
        cholesterol_hdl = st.slider("Kolesterol HDL", 20, 100, 50)
        cholesterol_ldl = st.slider("Kolesterol LDL", 50, 250, 120)
    with colE:
        blood_pressure_systolic = st.slider("Tekanan darah sistolik", 80, 200, 120)
        blood_pressure_diastolic = st.slider("Tekanan darah diastolik", 50, 130, 80)
        fasting_blood_sugar = st.slider("Gula darah puasa", 60, 300, 100)
    with colF:
        triglycerides = st.slider("Trigliserida", 50, 400, 150)
        waist_circumference = st.slider("Lingkar pinggang (cm)", 60, 150, 90)
        sleep_hours = st.slider("Rata-rata jam tidur", 3.0, 10.0, 7.0, step=0.1)

    input_row = pd.DataFrame([{
        "age": age, "cholesterol_level": cholesterol_level,
        "waist_circumference": waist_circumference, "sleep_hours": sleep_hours,
        "blood_pressure_systolic": blood_pressure_systolic,
        "blood_pressure_diastolic": blood_pressure_diastolic,
        "fasting_blood_sugar": fasting_blood_sugar, "cholesterol_hdl": cholesterol_hdl,
        "cholesterol_ldl": cholesterol_ldl, "triglycerides": triglycerides,
        "hypertension": int(hypertension), "diabetes": int(diabetes),
        "obesity": int(obesity), "family_history": int(family_history),
        "previous_heart_disease": int(previous_heart_disease),
        "medication_usage": int(medication_usage),
        "participated_in_free_screening": int(participated_in_free_screening),
        "gender": gender, "region": region, "income_level": income_level,
        "smoking_status": smoking_status, "alcohol_consumption": alcohol_consumption,
        "physical_activity": physical_activity, "dietary_habits": dietary_habits,
        "air_pollution_exposure": air_pollution_exposure, "stress_level": stress_level,
        "EKG_results": ekg_results,
    }])[model_bundle["features"]]

    if st.button("🔍 Hitung Prediksi", type="primary"):
        proba = pipe.predict_proba(input_row)[0, 1]
        pct = proba * 100

        if pct < 30:
            category, color = "Rendah", "green"
        elif pct < 60:
            category, color = "Sedang", "orange"
        else:
            category, color = "Tinggi", "red"

        # Disimpan di session_state (bukan cuma variabel lokal) supaya
        # hasilnya tetap ada setelah rerun yang dipicu tombol Groq di bawah
        # -- Streamlit rerun seluruh script tiap klik, dan st.button() biasa
        # otomatis balik ke False pada rerun yang bukan dia yang memicunya.
        st.session_state["prediction"] = {
            "pct": pct, "category": category, "color": color,
            "age": age, "gender": gender, "region": region, "income_level": income_level,
            "smoking_status": smoking_status, "alcohol_consumption": alcohol_consumption,
            "physical_activity": physical_activity, "dietary_habits": dietary_habits,
            "air_pollution_exposure": air_pollution_exposure, "stress_level": stress_level,
            "hypertension": hypertension, "diabetes": diabetes, "obesity": obesity,
            "previous_heart_disease": previous_heart_disease, "family_history": family_history,
            "medication_usage": medication_usage, "ekg_results": ekg_results,
            "blood_pressure_systolic": blood_pressure_systolic,
            "blood_pressure_diastolic": blood_pressure_diastolic,
            "cholesterol_level": cholesterol_level, "cholesterol_hdl": cholesterol_hdl,
            "cholesterol_ldl": cholesterol_ldl, "triglycerides": triglycerides,
            "fasting_blood_sugar": fasting_blood_sugar,
            "waist_circumference": waist_circumference, "sleep_hours": sleep_hours,
        }
        # Klik "Hitung Prediksi" baru berarti tulisan AI sebelumnya sudah
        # tidak relevan (punya profil lama), jadi dibuang.
        st.session_state.pop("ai_recommendation", None)

    # Dirender dari session_state (bukan cuma langsung setelah tombol)
    # supaya tetap tampil melewati rerun yang disebabkan tombol Groq.
    if "prediction" in st.session_state:
        r = st.session_state["prediction"]
        pct, category, color = r["pct"], r["category"], r["color"]

        colX, colY = st.columns([1, 2])
        with colX:
            st.metric("Estimasi probabilitas serangan jantung", f"{pct:.1f}%")
            st.markdown(f"**Kategori risiko: :{color}[{category}]**")
            st.caption(f"Prevalensi rata-rata populasi: {prevalence*100:.1f}%")

        with colY:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "#d4f4dd"},
                        {"range": [30, 60], "color": "#ffe8b3"},
                        {"range": [60, 100], "color": "#ffd0d0"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": prevalence * 100,
                    },
                },
                title={"text": "Skor Risiko (garis hitam = rata-rata populasi)"},
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.info(
            "Catatan: angka ini adalah **output model statistik** dari data sintetis, "
            "dipengaruhi terutama oleh riwayat sakit jantung, hipertensi, diabetes, "
            "obesitas, status merokok, dan usia. Bukan pengganti diagnosis dokter."
        )

        # -------------------------------------------------------------
        # Opsional: narasi klinis dari AI lewat Groq (Llama 3.3).
        # Dibuat sebagai tombol opt-in terpisah, bukan otomatis jalan --
        # ini memanggil API berbayar eksternal, menambah latency, dan
        # butuh GROQ_API_KEY, jadi tidak seharusnya menghalangi prediksi
        # ML utama di atas untuk tetap berfungsi sendiri.
        # -------------------------------------------------------------
        st.divider()
        st.markdown("### 🤖 Rekomendasi Klinis AI (opsional, via Groq)")

        if not GROQ_AVAILABLE:
            st.caption(
                "Fitur ini butuh package `groq`. Install dengan "
                "`pip install groq` untuk mengaktifkan narasi klinis dari AI."
            )
        else:
            try:
                api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
            except Exception:
                api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                st.caption(
                    "Fitur ini butuh `GROQ_API_KEY` (di-set sebagai environment "
                    "variable atau di `.streamlit/secrets.toml`) untuk membuat "
                    "narasi klinis dari AI berdasarkan angka-angka di atas."
                )
            elif st.button("Buat Rekomendasi Klinis AI"):
                system_prompt = """Anda adalah Senior Clinical Decision Support Specialist dan Analis Risiko Kardiovaskular. Tugas Anda adalah memberikan analisis risiko klinis yang personal dan berbasis bukti (evidence-based), beserta rekomendasi kesehatan, berdasarkan data pasien dan hasil dari model Machine Learning prediksi risiko.

Gaya Komunikasi & Aturan:
1. Nada & Bahasa: Profesional, empatik, dan menggunakan Bahasa Indonesia klinis yang berbasis bukti (evidence-based). Seluruh jawaban WAJIB dalam Bahasa Indonesia.
2. Struktur: Sajikan jawaban secara sistematis dengan format yang jelas (bullet point, judul tebal/bold).
3. Spesifisitas: Hindari saran generik (mis. "makan makanan sehat"). Kaitkan setiap rekomendasi langsung dengan nilai laboratorium, metrik klinis, atau faktor risiko spesifik pasien.
4. Disclaimer Medis: Selalu tutup jawaban dengan pemberitahuan jelas bahwa analisis ini adalah simulasi AI/CDSS dan harus dievaluasi oleh dokter spesialis jantung (kardiolog) atau tenaga medis bersertifikat."""

                user_prompt = f"""Berikan analisis klinis personal dan rekomendasi yang disesuaikan untuk profil pasien simulasi berikut ini:

=== PROFIL PASIEN & HASIL MODEL ML ===
- Probabilitas Risiko Serangan Jantung (Model ML): {pct:.1f}% (Kategori Risiko: {category})
- Prevalensi Rata-rata Populasi: {prevalence*100:.1f}%

Demografi & Gaya Hidup:
- Usia: {r['age']} | Gender: {r['gender']} | Wilayah: {r['region']} | Tingkat Pendapatan: {r['income_level']}
- Status Merokok: {r['smoking_status']} | Konsumsi Alkohol: {r['alcohol_consumption']} | Aktivitas Fisik: {r['physical_activity']} | Pola Makan: {r['dietary_habits']}
- Paparan Polusi Udara: {r['air_pollution_exposure']} | Tingkat Stres: {r['stress_level']}

Riwayat Medis & Komorbid:
- Hipertensi: {'Ya' if r['hypertension'] else 'Tidak'} | Diabetes: {'Ya' if r['diabetes'] else 'Tidak'} | Obesitas: {'Ya' if r['obesity'] else 'Tidak'}
- Riwayat Sakit Jantung Sebelumnya: {'Ya' if r['previous_heart_disease'] else 'Tidak'}
- Riwayat Keluarga: {'Ya' if r['family_history'] else 'Tidak'}
- Rutin Mengonsumsi Obat: {'Ya' if r['medication_usage'] else 'Tidak'} | Hasil EKG: {r['ekg_results']}

Metrik Klinis & Laboratorium:
- Tekanan Darah: {r['blood_pressure_systolic']}/{r['blood_pressure_diastolic']} mmHg
- Profil Lipid: Kolesterol Total {r['cholesterol_level']} mg/dL | HDL {r['cholesterol_hdl']} mg/dL | LDL {r['cholesterol_ldl']} mg/dL | Trigliserida {r['triglycerides']} mg/dL
- Gula Darah Puasa: {r['fasting_blood_sugar']} mg/dL | Lingkar Pinggang: {r['waist_circumference']} cm | Rata-rata Tidur: {r['sleep_hours']} jam/hari

=== STRUKTUR JAWABAN YANG DIHARAPKAN ===
Susun rekomendasi Anda ke dalam 4 bagian berikut (judul bagian dalam Bahasa Indonesia):

1. **Analisis Faktor Risiko Utama**
   - Identifikasi 3-4 parameter pasien di atas yang paling berisiko tinggi, dan jelaskan dampak patofisiologisnya terhadap sistem kardiovaskular.

2. **Langkah Klinis & Diagnostik Selanjutnya**
   - Rekomendasikan tes diagnostik lanjutan yang tepat sasaran (mis. Treadmill Exercise Test, Echocardiogram, HbA1c, ulang Panel Lipid) dan konsultasi spesialis yang sesuai.

3. **Intervensi Gaya Hidup yang Ditargetkan**
   - Berikan target kuantitatif (mis. target tekanan darah spesifik, target LDL, intensitas/durasi olahraga yang aman, kebiasaan tidur yang baik).

4. **Tanda Bahaya & Sinyal Peringatan**
   - Uraikan gejala akut kritis yang membutuhkan evaluasi Instalasi Gawat Darurat (IGD) segera (mis. nyeri dada menjalar, sesak napas mendadak)."""

                try:
                    with st.spinner("Membuat analisis klinis AI lewat Groq..."):
                        client = groq.Groq(api_key=api_key)
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.3,  # rendah: hasil deterministik, nada klinis
                            max_tokens=1500,
                        )
                    st.session_state["ai_recommendation"] = response.choices[0].message.content
                except Exception as e:
                    st.error(f"Permintaan ke Groq gagal: {e}")

            if "ai_recommendation" in st.session_state:
                st.markdown(st.session_state["ai_recommendation"])
