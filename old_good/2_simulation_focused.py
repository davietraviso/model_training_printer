import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load('2_random_forest_model.pkl')

# Pilih model printer secara manual
chosen_model = int(input("Masukkan nomor Model Printer (1-63): 200"))  # Input model printer

# Validasi input model
if chosen_model < 1 or chosen_model > 63:
    raise ValueError("Model Printer harus antara 1 hingga 63.")

# Jumlah data simulasi (maksimum 12 bulan)
num_simulations = 12

# Buat data simulasi dengan model yang dipilih
simulated_data = pd.DataFrame({
    'Month': np.arange(1, num_simulations + 1),  # Bulan 1-12
    'Year': [2024] * num_simulations,  # Tahun simulasi
    'Model': [chosen_model] * num_simulations,  # Hanya model yang dipilih
    'Merk_Brother': [1 if chosen_model in range(1, 9) else 0] * num_simulations,
    'Merk_Epson': [1 if chosen_model in range(9, 36) else 0] * num_simulations,
    'Merk_Canon': [1 if chosen_model in range(36, 51) else 0] * num_simulations,
    'Merk_HP': [1 if chosen_model in range(51, 64) else 0] * num_simulations
})

# Tambahkan kolom untuk kerusakan dan komponen (diisi 0 untuk default)
for i in range(1, 20):  # 19 kategori kerusakan
    simulated_data[f'Kerusakan_{i}'] = 0
for i in range(1, 21):  # 20 kategori komponen
    simulated_data[f'Komponen_{i}'] = 0

# Debug: Cek data simulasi yang dibuat
print("Data simulasi dibuat:")
print(simulated_data.head())

# Lakukan prediksi
predicted_values = model.predict(simulated_data)

# Gabungkan hasil prediksi ke data simulasi
simulated_data['Jumlah_Prediksi'] = predicted_values

# Simpan ke file
output_file = f'2_hasil_simulasi_model_{chosen_model}.xlsx'
simulated_data.to_excel(output_file, index=False)

print(f"Hasil prediksi untuk Model Printer {chosen_model} disimpan di '{output_file}'.")
print(simulated_data[['Month', 'Year', 'Model', 'Jumlah_Prediksi']])
