import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # Untuk menyimpan model (opsional)


# Baca data dari Excel
file_path = "raw.xlsx"  # Ganti dengan nama file Anda
data = pd.read_excel(file_path)

# Pisahkan fitur (input) dan target (output)
X = data.drop(columns=["Jumlah"])  # Semua kolom kecuali "Jumlah" adalah fitur
y = data["Jumlah"]  # Kolom "Jumlah" adalah target

# Split data ke training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Periksa hasil
print("Fitur setelah standarisasi (first 5 rows):")
print(X_train_scaled[:5])

# 1. Inisialisasi model Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 2. Training model
model.fit(X_train_scaled, y_train)

# 3. Prediksi dengan data testing
y_pred = model.predict(X_test_scaled)

# 4. Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 5. Cetak hasil evaluasi
print("Hasil Evaluasi Model:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# 6. Simpan model (opsional)
joblib.dump(model, "random_forest_model.pkl")
