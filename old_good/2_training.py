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
model = RandomForestRegressor(random_state=42, n_estimators=100)  # 100 pohon
model.fit(X_train, y_train)

# 5. Prediksi pada data uji
y_pred = model.predict(X_test)

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
predictions.to_excel("2_predictions.xlsx", index=False)
print("\nPrediksi disimpan ke '2_predictions.xlsx'.")
