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

# 5. Periksa hasil standarisasi (opsional)
print("Fitur setelah standarisasi (first 5 rows):")
print(X_train_scaled[:5])

# 6. Hyperparameter Tuning dengan GridSearchCV
# Definisi parameter yang akan diuji
param_grid = {
    'n_estimators': [100, 200, 300],         # Jumlah pohon
    'max_depth': [10, 20, None],            # Kedalaman maksimum
    'min_samples_split': [2, 5, 10],        # Minimum split
    'min_samples_leaf': [1, 2, 4],          # Minimum leaf
}

# Inisialisasi model Random Forest
rf = RandomForestRegressor(random_state=42)

# GridSearchCV untuk mencari parameter terbaik
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',  # Evaluasi menggunakan MAE
    cv=3,                               # 3-fold cross-validation
    verbose=2,
    n_jobs=-1                           # Paralel untuk efisiensi
)

# 7. Training model dengan GridSearchCV
print("Melakukan hyperparameter tuning...")
grid_search.fit(X_train_scaled, y_train)

# Parameter terbaik
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 8. Prediksi dengan data testing
y_pred = best_model.predict(X_test_scaled)

# 9. Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nHasil Evaluasi Model Setelah Hyperparameter Tuning:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# 10. Simpan model terbaik (opsional)
joblib.dump(best_model, "optimized_random_forest_model.pkl")