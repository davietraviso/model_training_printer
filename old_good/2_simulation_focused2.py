import numpy as np
import pandas as pd
import joblib

# Load model dan daftar kolom
model = joblib.load('2_random_forest_model.pkl')
X_train_columns = joblib.load('2_train_columns.pkl')

# Input model printer
chosen_model = input("Masukkan nomor Model Printer (1-63) (default=3): ")
chosen_model = int(chosen_model) if chosen_model else 3

if chosen_model < 1 or chosen_model > 63:
    raise ValueError("Model Printer harus antara 1 hingga 63.")

# Buat data simulasi
num_simulations = 12
simulated_data = pd.DataFrame({
    'Month': np.arange(1, num_simulations + 1),
    'Year': [2024] * num_simulations,
    'Model': [chosen_model] * num_simulations,
    'Merk_Brother': [1 if chosen_model in range(1, 9) else 0] * num_simulations,
    'Merk_Epson': [1 if chosen_model in range(9, 36) else 0] * num_simulations,
    'Merk_Canon': [1 if chosen_model in range(36, 51) else 0] * num_simulations,
    'Merk_HP': [1 if chosen_model in range(51, 64) else 0] * num_simulations
})

# Tambahkan One-Hot Encoding
for i in range(20):
    simulated_data[f'Kerusakan_{i}'] = 0
for i in range(21):
    simulated_data[f'Komponen_{i}'] = 0

# Validasi kolom
missing_columns = set(X_train_columns) - set(simulated_data.columns)
for col in missing_columns:
    simulated_data[col] = 0
simulated_data = simulated_data[X_train_columns]

# Prediksi
predicted_values = model.predict(simulated_data)

# Simpan hasil
simulated_data['Jumlah_Prediksi'] = predicted_values
output_file = f'2_hasil_simulasi_model_{chosen_model}.xlsx'
simulated_data.to_excel(output_file, index=False)

print(f"Hasil prediksi untuk Model Printer {chosen_model} disimpan di '{output_file}'.")
