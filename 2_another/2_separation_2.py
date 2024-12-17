import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. Baca file Excel
file_path = "2_raw.xlsx"  # Ganti nama file sesuai file Anda
df = pd.read_excel(file_path)

# 2. Encoding kolom 'Month' dengan Sine-Cosine Encoding
month_mapping = {
    "January": 0, "February": 1, "March": 2, "April": 3, "May": 4, "June": 5,
    "July": 6, "August": 7, "September": 8, "October": 9, "November": 10, "December": 11
}
df["Month"] = df["Month"].map(month_mapping)
df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)

# 3. Encoding kolom 'Year' (2018-2023 -> 0-5)
year_mapping = {2018: 0, 2019: 1, 2020: 2, 2021: 3, 2022: 4, 2023: 5}
df["Year"] = df["Year"].map(year_mapping)

# 4. Pastikan tipe data string untuk kolom 'Model', 'Kerusakan', dan 'Komponen'
df["Model"] = df["Model"].astype(str)
df["Kerusakan"] = df["Kerusakan"].astype(str)
df["Komponen"] = df["Komponen"].astype(str)

# 5. Label Encoding untuk kolom 'Model', 'Kerusakan', dan 'Komponen'
label_encoder = LabelEncoder()
df["Model"] = label_encoder.fit_transform(df["Model"])
df["Kerusakan"] = label_encoder.fit_transform(df["Kerusakan"])
df["Komponen"] = label_encoder.fit_transform(df["Komponen"])

# 6. One-Hot Encoding untuk kolom 'Merk', 'Kerusakan', dan 'Komponen'
df = pd.get_dummies(df, columns=["Merk", "Kerusakan", "Komponen"], prefix=["Merk", "Kerusakan", "Komponen"])

# ----- Pembagian Dataset berdasarkan Model dan Year -----
train_list = []
test_list = []

# Dapatkan list unik model
models = df["Model"].unique()

for model in models:
    model_df = df[df["Model"] == model]  # Filter berdasarkan model
    # Pisahkan data training (Year 0-4) dan testing (Year 5)
    train_data = model_df[model_df["Year"] <= 4]
    test_data = model_df[model_df["Year"] == 5]
    
    # Tambahkan ke daftar
    train_list.append(train_data)
    test_list.append(test_data)

# Gabungkan semua hasil pemisahan
train_df = pd.concat(train_list)
test_df = pd.concat(test_list)

# Pisahkan fitur (X) dan target (y)
X_train = train_df.drop(columns=["Jumlah"])
y_train = train_df["Jumlah"]
X_test = test_df.drop(columns=["Jumlah"])
y_test = test_df["Jumlah"]

# ----- Menyimpan Dataset yang Sudah Diproses ke File Excel -----
# Menggabungkan X_train dan y_train menjadi satu DataFrame
train_df = pd.concat([X_train, y_train], axis=1)
train_df.to_excel("./2_another/output/5_train_data.xlsx", index=False)

# Menggabungkan X_test dan y_test menjadi satu DataFrame
test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_excel("./2_another/output/5_test_data.xlsx", index=False)

print("Dataset berhasil dibagi berdasarkan model dan year, dan disimpan ke file Excel.")
