import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # Untuk menyimpan model (opsional)

# 1. Baca data dari Excel
file_path = "raw.xlsx"  # Ganti dengan nama file Anda
data = pd.read_excel(file_path)

# 2. Pisahkan fitur (input) dan target (output)
X = data.drop(columns=["Jumlah"])  # Semua kolom kecuali "Jumlah" adalah fitur
y = data["Jumlah"]  # Kolom "Jumlah" adalah target

# 3. Split data ke training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standarisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Hyperparameter Tuning dengan GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],         # Jumlah pohon
    'max_depth': [10, 20, None],            # Kedalaman maksimum
    'min_samples_split': [2, 5, 10],        # Minimum split
    'min_samples_leaf': [1, 2, 4],          # Minimum leaf
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',  # Evaluasi menggunakan MAE
    cv=3,                               # 3-fold cross-validation
    verbose=2,
    n_jobs=-1                           # Paralel untuk efisiensi
)

print("Melakukan hyperparameter tuning...")
grid_search.fit(X_train_scaled, y_train)

# 6. Ambil model terbaik dari hasil GridSearchCV
best_model = grid_search.best_estimator_

# Cetak parameter terbaik
print("\nBest Parameters:", grid_search.best_params_)

# 7. Evaluasi model terbaik pada testing set
y_pred = best_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 1. Model tanpa Hyperparameter Tuning (default)
rf_default = RandomForestRegressor(random_state=42)
rf_default.fit(X_train_scaled, y_train)
y_pred_default = rf_default.predict(X_test_scaled)

mae_default = mean_absolute_error(y_test, y_pred_default)
mse_default = mean_squared_error(y_test, y_pred_default)
r2_default = r2_score(y_test, y_pred_default)

print("\nHasil Evaluasi Model Tanpa Hyperparameter Tuning:")
print("Mean Absolute Error (MAE):", mae_default)
print("Mean Squared Error (MSE):", mse_default)
print("R-squared (R²):", r2_default)

print("")

print("\nHasil Evaluasi Model Setelah Hyperparameter Tuning:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# 8. Simpan model terbaik (opsional)
joblib.dump(best_model, "optimized_random_forest_model.pkl")
