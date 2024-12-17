import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1. Baca file Excel
file_path = "./another/2_raw.xlsx"  # Ganti nama file sesuai file Anda
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

# 7. Siapkan fitur (X) dan target (y)
X = df.drop(columns=["Jumlah"])  # Fitur adalah semua kolom kecuali 'Jumlah'
y = df["Jumlah"]  # Target adalah kolom 'Jumlah'

# 8. Tampilkan hasil preprocessing
print("Hasil Preprocessing:")
print(df.head())

# 9. Simpan hasil preprocessing ke file baru (opsional)
output_file = "./another/output/3_preprocessed.xlsx"
df.to_excel(output_file, index=False)
print(f"Hasil preprocessing telah disimpan ke: {output_file}")
