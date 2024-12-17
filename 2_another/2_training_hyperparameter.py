# 1. Import library
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pandas as pd

# 2. Load dataset
train_data = pd.read_excel("./another/3_train_data.xlsx")  # Data latih
test_data = pd.read_excel("./another/3_test_data.xlsx")    # Data uji

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
# predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
# predictions.to_excel("2_predictions.xlsx", index=False)
# print("\nPrediksi disimpan ke '2_predictions.xlsx'.")

# Parameter yang akan diuji pada Grid Search
param_grid = {
    'n_estimators': [50, 100, 150],       # Jumlah pohon
    'max_depth': [None, 10, 20, 30],      # Kedalaman pohon
    'min_samples_split': [2, 5, 10],      # Minimum sampel untuk split
    'min_samples_leaf': [1, 2, 4]         # Minimum sampel di setiap daun
}

# Inisialisasi model Random Forest
rf_model = RandomForestRegressor(random_state=42)

# Grid Search dengan Cross Validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)

# Melakukan Grid Search pada data latih
print("Memulai tuning hyperparameter...")
grid_search.fit(X_train, y_train)

# Parameter terbaik
best_params = grid_search.best_params_
print(f"Parameter terbaik: {best_params}")

# Evaluasi model dengan parameter terbaik
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluasi ulang dengan parameter terbaik
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)

print("\nHasil Evaluasi Model dengan Parameter Terbaik:")
print(f"Mean Absolute Error (MAE): {mae_best:.4f}")
print(f"Mean Squared Error (MSE): {mse_best:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_best:.4f}")
print(f"R-squared (R²): {r2_best:.4f}")

# Simpan hasil prediksi baru ke file Excel
# predictions_best_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_best})
# predictions_best_df.to_excel("2_predictions_best.xlsx", index=False)
# print("Prediksi terbaik disimpan ke file: 2_predictions_best.xlsx")

# Simpan model ke file
joblib.dump(model, './another/output/3_random_forest_model.pkl')

# Simpan fitur ke file
X_train.to_csv('./another/output/3_X_train.csv', index=False)