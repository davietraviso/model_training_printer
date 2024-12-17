import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Contoh data mentah
data = {
    "Kerusakan": ["USB PATAH", "TAK BISA PRINT", "GANTI HEAD", "BURAM", "ERROR"],
    "Komponen": ["USB PORT", "DRUM PRINTER", "PRINT HEAD", "INK CARTRIDGE", "BELTING"],
    "Model": ["DCP-J100", "DCP-T300", "L3110", "MFC-J200", "PIXMA MG2570"]
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Inisialisasi LabelEncoder untuk setiap kolom
label_encoder_kerusakan = LabelEncoder()
label_encoder_komponen = LabelEncoder()
label_encoder_model = LabelEncoder()

# Melakukan encoding pada kolom-kolom yang diperlukan
df["Kerusakan_Encoded"] = label_encoder_kerusakan.fit_transform(df["Kerusakan"])
df["Komponen_Encoded"] = label_encoder_komponen.fit_transform(df["Komponen"])
df["Model_Encoded"] = label_encoder_model.fit_transform(df["Model"])

# Membuat mapping hasil encoding ke nilai aslinya
kerusakan_mapping = pd.DataFrame({
    "Kerusakan": label_encoder_kerusakan.classes_,
    "Encoded": range(len(label_encoder_kerusakan.classes_))
})

komponen_mapping = pd.DataFrame({
    "Komponen": label_encoder_komponen.classes_,
    "Encoded": range(len(label_encoder_komponen.classes_))
})

model_mapping = pd.DataFrame({
    "Model": label_encoder_model.classes_,
    "Encoded": range(len(label_encoder_model.classes_))
})

# Menampilkan hasil mapping
print("Mapping Kerusakan:")
print(kerusakan_mapping)
print("\nMapping Komponen:")
print(komponen_mapping)
print("\nMapping Model:")
print(model_mapping)
