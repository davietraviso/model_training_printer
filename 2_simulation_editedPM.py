import pandas as pd
import numpy as np
import joblib
import os

# Daftar kolom X_train dari file
X_train_columns = joblib.load('2_train_columns.pkl')

# Tahap 2: Membuat data simulasi
num_samples = 12  # Misalnya, 12 data untuk setiap bulan di tahun 2024
simulated_data = pd.DataFrame({
    "Month": np.tile(np.arange(1, 13), 1),  # Bulan 1 hingga 12
    "Year": [2024] * num_samples,
    "Model": [10] * num_samples,  # Simulasi model printer
    "Jumlah": [0] * num_samples  # Default 0 (bisa diganti sesuai kebutuhan)
})

# Sine-Cosine encoding untuk kolom bulan
simulated_data["Month_Sin"] = np.sin(2 * np.pi * simulated_data["Month"] / 12)
simulated_data["Month_Cos"] = np.cos(2 * np.pi * simulated_data["Month"] / 12)

# One-Hot Encoding untuk Merk
for merk in ["Merk_Brother", "Merk_Canon", "Merk_Epson", "Merk_HP"]:
    simulated_data[merk] = np.random.choice([0, 1], size=num_samples)

kerusakan_aktif = [5]  # Daftar kerusakan yang aktif
komponen_aktif = [13]  # Daftar komponen yang aktif

# Setel default 0 untuk semua kerusakan dan komponen
for i in range(20):  # Total 20 jenis kerusakan
    simulated_data[f"Kerusakan_{i}"] = 0
for i in range(21):  # Total 21 jenis komponen
    simulated_data[f"Komponen_{i}"] = 0

# Aktifkan kerusakan dan komponen yang relevan
for kerusakan in kerusakan_aktif:
    simulated_data[f"Kerusakan_{kerusakan}"] = 1
for komponen in komponen_aktif:
    simulated_data[f"Komponen_{komponen}"] = 1

# Gabungkan kolom asli dengan kolom yang sesuai untuk prediksi
X_simulated = simulated_data[X_train_columns]

# Simpan data simulasi ke file CSV (opsional)
X_simulated.to_csv('2_simulated_data.csv', index=False)

print("Data simulasi dibuat:")
print(X_simulated.head())

# Tahap 3: Prediksi
# Load model yang telah dilatih
model = joblib.load('2_random_forest_model.pkl')

# Lakukan prediksi
predicted_values = model.predict(X_simulated)

# Gabungkan hasil prediksi dengan data simulasi
simulated_data['Jumlah_Prediksi'] = predicted_values

# Simpan hasil prediksi ke file Excel
# Path file awal
base_filename = './testing/2_hasil_simulasi_prediksi_'

# Cek apakah file sudah ada, jika ada, tambahkan angka increment
counter = 0
while os.path.exists(f"{base_filename}{counter}.xlsx"):
    counter += 1

# Simpan data ke file dengan nama yang sudah terincrement
simulated_data.to_excel(f"{base_filename}{counter}.xlsx", index=False)

print("Hasil prediksi disimpan di './testing/2_hasil_simulasi_prediksi_X.xlsx'.")

print("Hasil prediksi:")
print(simulated_data[["Month", "Year", "Model", "Jumlah_Prediksi"]])


# Membaca dataset mentah
# data = pd.read_csv('data_mentah.csv')

# # One-Hot Encoding untuk Merk
# data = pd.get_dummies(data, columns=['Merk'])

# # One-Hot Encoding untuk Kerusakan
# for i in range(20):  # Total 20 jenis kerusakan
#     data[f"Kerusakan_{i}"] = (data['Damage'] == f'KERUSAKAN_{i}').astype(int)

# # Sine-Cosine Encoding untuk Bulan
# data["Month_Sin"] = np.sin(2 * np.pi * data["Month"] / 12)
# data["Month_Cos"] = np.cos(2 * np.pi * data["Month"] / 12)

# # Memisahkan fitur dan target
# X = data.drop(columns=['Jumlah', 'Komponen'])  # Fitur
# y = data[['Jumlah', 'Komponen']]  # Target

# # Menyimpan data yang telah diproses
# X.to_csv('2_train_data_preprocessed.csv', index=False)