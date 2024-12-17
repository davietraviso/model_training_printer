import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
