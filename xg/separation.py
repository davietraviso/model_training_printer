import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Import dataset dari file
file_path = '2_preprocessed.xlsx'  # Ganti dengan path file Anda
dataset = pd.read_excel(file_path)

# 2. Pisahkan fitur (X) dan target (y)
X = dataset.drop(columns=["Jumlah"])  # Semua kolom kecuali target
y = dataset["Jumlah"]  # Target kolom

# 3. Bagi dataset menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Simpan hasil split ke file
X_train.to_excel('./xg/output/X_train.xlsx', index=False)
X_test.to_excel('./xg/output/X_test.xlsx', index=False)
y_train.to_excel('./xg/output/y_train.xlsx', index=False)
y_test.to_excel('./xg/output/y_test.xlsx', index=False)

print("Dataset berhasil dipisah dan disimpan ke folder './output'")
