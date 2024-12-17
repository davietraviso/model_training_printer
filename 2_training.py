# 1. Import library
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load dataset
train_data = pd.read_excel("2_train_data.xlsx")  # Data latih
test_data = pd.read_excel("2_test_data.xlsx")    # Data uji

# 3. Pisahkan fitur (X) dan target (y)
X_train = train_data.drop("Jumlah", axis=1)  # Semua kolom kecuali "Jumlah"
y_train = train_data["Jumlah"]               # Kolom target
X_test = test_data.drop("Jumlah", axis=1)
y_test = test_data["Jumlah"]

# 4. Inisialisasi dan latih model Random Forest
model = RandomForestRegressor(random_state=62, n_estimators=350)  # 100 pohon
model.fit(X_train, y_train)

# 5. Prediksi pada data uji
y_pred = model.predict(X_test)

# Hitung manual
# mae = np.mean(np.abs(y_test - y_pred))
# print("MAE:", mae)

# mse = np.mean((y_test - y_pred) ** 2)
# print("MSE:", mse)

# rmse = np.sqrt(mse)
# print("RMSE:", rmse)

# r2 = r2_score(y_test, y_pred)
# print("R-squared:", r2)


# 6. Evaluasi performa model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 7. Cetak hasil evaluasi
print("Hasil Evaluasi Model:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# 8. Simpan prediksi ke file Excel (opsional)
predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
# predictions.to_excel("2_predictions_2.xlsx", index=False)
print("\nPrediksi disimpan ke '2_predictions_2.xlsx'.")

# 9. Buat tabel hasil representasi
results_df = pd.DataFrame({
    "y_test (Actual)": y_test.values,  # Nilai Aktual
    "y_pred (Predicted)": y_pred,      # Nilai Prediksi
})

# Hitung kolom tambahan
results_df["Error (y_test - y_pred)"] = results_df["y_test (Actual)"] - results_df["y_pred (Predicted)"]
results_df["Absolute Error"] = results_df["Error (y_test - y_pred)"].abs()
results_df["Squared Error"] = results_df["Error (y_test - y_pred)"] ** 2

# Ambil hanya 5 sampel untuk representasi
# sample_results = results_df.head(25)  # Ambil 5 baris pertama

# Cetak hasil tabel
print("\nRepresentasi Hasil Prediksi (5 Sampel):")
# print(sample_results.to_string(index=False))  # Tampilkan tanpa indeks

# 1. Hitung rata-rata dari y_test (nilai aktual)
y_test_mean = np.mean(y_test)

# 2. Hitung Total Sum of Squares (TSS)
tss = np.sum((y_test - y_test_mean) ** 2)

# 3. Hitung Residual Sum of Squares (RSS)
rss = np.sum((y_test - y_pred) ** 2)

# 4. Hitung R-squared (R²)
r2_manual = 1 - (rss / tss)

# 5. Cetak R² manual

print("Beberapa Nilai y_test (Actual):", y_test.values[:25])
print("Beberapa Nilai y_pred (Predicted):", y_pred[:25])
print(f"Y_Test mean: {y_test_mean:.4f}")
print(f"TSS: {tss:.4f}")
print(f"RSS: {rss:.4f}")

print(f"R-squared (R²) Manual: {r2_manual:.4f}")
