import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. Baca file Excel
file_path = "./2_another/3_raw.xlsx"  # Ganti nama file sesuai file Anda
df = pd.read_excel(file_path)

# 2. Encoding kolom 'Month' dengan Sine-Cosine Encoding
# Map bulan ke angka 0-11
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

# ----- Pembagian Dataset -----
X = df.drop('Jumlah', axis=1)  # Semua fitur selain target 'Jumlah'
y = df['Jumlah']  # Target variabel 'Jumlah'

# Membagi dataset menjadi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menampilkan ukuran data latih dan uji
print(f"Ukuran data latih: {X_train.shape[0]} baris")
print(f"Ukuran data uji: {X_test.shape[0]} baris")

# ----- Menyimpan Dataset yang Sudah Diproses ke File Excel -----
# Menggabungkan X_train dan y_train menjadi satu DataFrame
train_df = pd.concat([X_train, y_train], axis=1)
train_df.to_excel("./2_another/output/3_train_data.xlsx", index=False)

# Simpan kolom X_train ke file
# import joblib
# joblib.dump(X_train.columns.tolist(), '2_train_columns.pkl')

# Menggabungkan X_test dan y_test menjadi satu DataFrame
test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_excel("./2_another/output/3_test_data.xlsx", index=False)

print("Dataset berhasil dibagi dan disimpan ke file Excel.")
