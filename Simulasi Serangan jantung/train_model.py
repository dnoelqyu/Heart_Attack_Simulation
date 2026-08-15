"""
Membangun model prediksi risiko heart attack dari dataset yang SUDAH
DIBERSIHKAN di Orange (Step 1: File -> Impute -> Save Data), lalu
menyimpan model + metadata supaya bisa dipakai ulang di dashboard (tanpa
retrain tiap kali dashboard dibuka).

Arsitektur pembersihan data:
- Pembersihan (isi nilai kosong alcohol_consumption dll) HANYA terjadi di
  Orange, lewat widget Impute -> Save Data, diekspor sebagai
  heart_attack_cleaned.csv.
- Script ini TIDAK melakukan pembersihan apa pun -- tidak ada fillna(),
  tidak ada logic imputasi. Ia murni memuat file yang sudah bersih dari
  Orange. Kalau file itu belum ada, script berhenti dengan pesan jelas
  (bukan diam-diam membersihkan sendiri) -- supaya arsitektur "bersih
  cuma di satu tempat" ini benar-benar ditegakkan, bukan cuma niat baik.

Arsitektur pemilihan algoritma:
- Perbandingan algoritma (Logistic Regression, Random Forest, Gradient
  Boosting, kNN, Naive Bayes) HANYA terjadi di Orange (Step 1), lewat
  Test and Score. Gradient Boosting terpilih sebagai yang paling efektif.
- Script ini TIDAK membandingkan kandidat lagi -- ia langsung membangun
  DAN melatih Gradient Boosting saja, sesuai keputusan yang sudah diambil.
  Tidak ada LogisticRegression atau RandomForestClassifier di sini.

Jalankan sekali (setelah heart_attack_cleaned.csv diekspor dari Orange):
    python train_model.py
Output: heart_attack_model.joblib
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLEANED_FILE = "heart_attack_cleaned.csv"  # exported from Orange: File -> Impute -> Save Data

if not os.path.exists(CLEANED_FILE):
    raise FileNotFoundError(
        f"\n\n'{CLEANED_FILE}' tidak ditemukan.\n"
        f"Script ini sengaja TIDAK membersihkan data sendiri -- pembersihan\n"
        f"hanya boleh terjadi di Orange (Step 1). Langkah yang perlu dilakukan:\n"
        f"  1. Buka workflow Orange, sambungkan File -> Impute -> Save Data\n"
        f"  2. Set nama file output Save Data ke '{CLEANED_FILE}'\n"
        f"  3. Taruh file itu di folder yang sama dengan script ini\n"
        f"  4. Jalankan ulang: python train_model.py\n"
    )

print(f"Memuat data yang sudah dibersihkan di Orange: {CLEANED_FILE}")
# keep_default_na=False penting: tanpa ini, pandas otomatis menganggap teks
# "None" (kategori sah untuk alcohol_consumption) sebagai nilai kosong saat
# dibaca ulang -- membatalkan pembersihan yang sudah dilakukan di Orange.
# na_values=[""] tetap menandai sel yang BENAR-BENAR kosong sebagai NaN.
df = pd.read_csv(CLEANED_FILE, keep_default_na=False, na_values=[""], low_memory=False)

# Orange's Save Data widget sometimes writes 2 extra rows right after the
# header: a domain/type descriptor row (e.g. "continuous", "0 1",
# "Female Male") and a blank role row -- both are Orange's internal format
# metadata, not patient data. Detect them by finding the first row where
# "age" actually parses as a number, and drop everything before it. This
# adapts automatically whether Orange includes 0, 1, or 2 such rows.
age_numeric = pd.to_numeric(df["age"], errors="coerce")
first_valid = age_numeric.first_valid_index()
if first_valid is not None and first_valid > 0:
    print(f"Melewati {first_valid} baris metadata Orange di awal file (bukan data pasien).")
    df = df.iloc[first_valid:].reset_index(drop=True)

TARGET = "heart_attack"
NUMERIC = [
    "age", "cholesterol_level", "waist_circumference", "sleep_hours",
    "blood_pressure_systolic", "blood_pressure_diastolic", "fasting_blood_sugar",
    "cholesterol_hdl", "cholesterol_ldl", "triglycerides",
]
BINARY = [
    "hypertension", "diabetes", "obesity", "family_history",
    "previous_heart_disease", "medication_usage", "participated_in_free_screening",
]
CATEGORICAL = [
    "gender", "region", "income_level", "smoking_status", "alcohol_consumption",
    "physical_activity", "dietary_habits", "air_pollution_exposure",
    "stress_level", "EKG_results",
]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# Defensive: force numeric columns to actually be numeric. Some Orange CSV
# exports include an extra metadata row (column type markers) right after
# the header -- if that slips into the data, a single stray text value
# forces pandas to treat the whole column as text, which breaks
# StandardScaler later with a confusing error. errors="coerce" turns any
# value that isn't a real number into NaN, which the check below then
# catches with a clear message instead.
for col in NUMERIC:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

# Sanity check: this script trusts Orange did the cleaning, but let's verify
# that trust rather than assume it blindly. If the Impute widget in Orange
# was misconfigured (e.g. wrong strategy, or a column left unconnected),
# this catches it here with a clear message -- not as a confusing crash
# deep inside sklearn later.
missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"\n\nKolom berikut tidak ada di '{CLEANED_FILE}': {missing_cols}\n"
        f"Cek widget Select Columns / Save Data di Orange -- pastikan semua\n"
        f"kolom ini disertakan saat ekspor.\n"
    )

null_counts = df[FEATURES + [TARGET]].isna().sum()
still_null = null_counts[null_counts > 0]
if len(still_null):
    raise ValueError(
        f"\n\n'{CLEANED_FILE}' masih punya nilai kosong setelah 'dibersihkan' di Orange:\n"
        f"{still_null.to_string()}\n\n"
        f"Ini tandanya widget Impute belum dikonfigurasi untuk kolom itu (atau\n"
        f"strateginya tidak menutupi semua baris kosong). Perbaiki di Orange,\n"
        f"ekspor ulang, lalu jalankan lagi -- script ini sengaja tidak menebak\n"
        f"nilai kosong sendiri.\n"
    )
print(f"Validasi lolos: {len(df):,} baris, {len(FEATURES)} fitur, tidak ada nilai kosong.")

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ]
)

# Only Gradient Boosting is built here -- the algorithm choice was already
# made in Orange (Step 1, comparing 5 candidates). Python does not
# re-compare candidates; it implements the one that was selected.
pipe = Pipeline([
    ("prep", preprocess),
    ("clf", GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.9, random_state=42,
    )),
])

pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print("\n=== gradient_boosting ===")
print(f"ROC-AUC (data uji): {auc:.4f}")
print(f"Prevalensi heart_attack di data: {y.mean()*100:.2f}%")
preds = (proba >= 0.5).astype(int)
print(classification_report(y_test, preds, digits=3))
print("Confusion matrix [ [TN FP] [FN TP] ]:")
print(confusion_matrix(y_test, preds))

best_name = "gradient_boosting"
best_pipe = pipe
print(f"\nModel: {best_name} (AUC={auc:.4f}) -- satu-satunya kandidat yang dibangun di sini")

# Feature importance untuk narasi presentasi
feature_names = (
    NUMERIC + BINARY
    + list(best_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
)
importances = best_pipe.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"fitur": feature_names, "skor": importances}).sort_values(
    "skor", ascending=False
)
print("\nTop 15 fitur paling berpengaruh:")
print(imp_df.head(15).to_string(index=False))

joblib.dump(
    {
        "pipeline": best_pipe,
        "model_name": best_name,
        "auc": auc,
        "features": FEATURES,
        "numeric": NUMERIC,
        "binary": BINARY,
        "categorical": CATEGORICAL,
        "prevalence": float(y.mean()),
        "top_features": imp_df.head(15).to_dict(orient="records"),
    },
    "heart_attack_model.joblib",
)
with open("model_meta.json", "w") as f:
    json.dump(
        {
            "model_name": best_name,
            "auc": auc,
            "prevalence": float(y.mean()),
            "top_features": imp_df.head(15).to_dict(orient="records"),
        },
        f,
        indent=2,
    )
print("\nModel disimpan: heart_attack_model.joblib")
