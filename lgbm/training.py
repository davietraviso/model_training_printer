import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Baca file Excel (menggunakan data yang sudah diproses sebelumnya)
file_path = "2_preprocessed.xlsx"  # Ganti sesuai file yang sudah digabung
df = pd.read_excel(file_path)

# 2. Pisahkan fitur (X) dan target (y)
X = df.drop('Jumlah', axis=1)  # Semua fitur selain 'Jumlah'
y = df['Jumlah']  # Target adalah 'Jumlah'

# 3. Pembagian data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pelatihan Model Dasar LGBM
lgbm_model = lgb.LGBMRegressor(n_estimators=100)  # Model dasar dengan 100 estimators
lgbm_model.fit(X_train, y_train)

# 5. Prediksi pada data uji
y_pred = lgbm_model.predict(X_test)

# 6. Evaluasi Model: R² dan MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# 7. Output hasil evaluasi
print("Evaluasi Model Dasar LGBM:")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# Opsional: Menampilkan beberapa nilai prediksi dan nilai asli
pred_comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nPerbandingan Prediksi vs Actual:")
print(pred_comparison.head())
