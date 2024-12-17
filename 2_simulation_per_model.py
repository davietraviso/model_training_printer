import pandas as pd
import numpy as np
import joblib
import os

# Daftar kolom X_train dari file
X_train_columns = joblib.load('2_train_columns.pkl')

# Tahap 2: Membuat data simulasi per bulan untuk model printer tertentu
num_samples = 12  # Simulasi untuk 12 bulan dalam tahun 2024
model_id = 10  # Misalnya kita pilih model 10

# Menginisialisasi DataFrame untuk simulasi
simulated_data = pd.DataFrame({
    "Month": np.tile(np.arange(1, 13), 1),  # Bulan 1 hingga 12
    "Year": [2024] * num_samples,
    "Model": [model_id] * num_samples,  # Model printer 10
    "Jumlah": [0] * num_samples  # Default 0, akan diupdate berdasarkan simulasi
})

# Sine-Cosine encoding untuk kolom bulan
simulated_data["Month_Sin"] = np.sin(2 * np.pi * simulated_data["Month"] / 12)
simulated_data["Month_Cos"] = np.cos(2 * np.pi * simulated_data["Month"] / 12)

# One-Hot Encoding untuk Merk
for merk in ["Merk_Brother", "Merk_Canon", "Merk_Epson", "Merk_HP"]:
    simulated_data[merk] = np.random.choice([0, 1], size=num_samples)

# Simulasikan Kerusakan (Kerusakan_0 hingga Kerusakan_19) berdasarkan model
for i in range(20):  # Total 20 jenis kerusakan
    simulated_data[f"Kerusakan_{i}"] = 0

# Misalnya, model 10 memiliki kerusakan 11 sebesar 100%
if model_id == 10:
    simulated_data.loc[:, "Kerusakan_11"] = 1  # Kerusakan_11 aktif

# One-Hot Encoding untuk Komponen (Komponen_0 hingga Komponen_20) berdasarkan model
for i in range(21):  # Total 21 jenis komponen
    simulated_data[f"Komponen_{i}"] = 0

# Misalnya, model 10 membutuhkan komponen 18 sebesar 100%
if model_id == 10:
    simulated_data.loc[:, "Komponen_18"] = 1  # Komponen_18 aktif

# Gabungkan kolom asli dengan kolom yang sesuai untuk prediksi
X_simulated = simulated_data[X_train_columns]

# Simpan data simulasi ke file CSV (opsional)
# X_simulated.to_csv('2_simulated_data.csv', index=False)

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

print("Hasil prediksi disimpan di './testing/2_hasil_simulasi_prediksi.xlsx'.")
print("Hasil prediksi:")
print(simulated_data[["Month", "Year", "Model", "Jumlah_Prediksi"]])
