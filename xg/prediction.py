import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset hasil split
X_train = pd.read_excel('./xg/output/X_train.xlsx')
X_test = pd.read_excel('./xg/output/X_test.xlsx')
y_train = pd.read_excel('./xg/output/y_train.xlsx')
y_test = pd.read_excel('./xg/output/y_test.xlsx')

# 2. Inisialisasi model XGBoost
model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Error kuadrat untuk regresi
    n_estimators=300,             # Jumlah pohon
    learning_rate=0.05,            # Kecepatan belajar
    max_depth=4,                  # Kedalaman maksimal pohon
    random_state=42
)

# 3. Latih model dengan data training
model.fit(X_train, y_train)

# 4. Prediksi dengan data testing
y_pred = model.predict(X_test)

# 5. Evaluasi performa model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Hasil Evaluasi Model:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
# 6. Simpan hasil prediksi ke file
y_pred_df = pd.DataFrame(y_pred, columns=["Prediction"])
# y_pred_df.to_excel('./xg/output/y_pred_2.xlsx', index=False)

print("Hasil prediksi disimpan ke './xg/output/y_pred.xlsx'")
