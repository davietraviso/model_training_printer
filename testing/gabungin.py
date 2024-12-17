import pandas as pd

# Folder tempat file berada
folder_path = './testing/'  # Ganti dengan folder lokasi file

# Format nama file dan range angka (0 hingga 112)
file_paths = [f"{folder_path}2_hasil_simulasi_prediksi_{i}.xlsx" for i in range(113)]

# List untuk menyimpan data dari setiap file
data_frames = []

# Loop melalui setiap file, membaca isinya, lalu menambahkannya ke list
for file in file_paths:
    try:
        df = pd.read_excel(file)  # Baca file Excel
        data_frames.append(df)    # Tambahkan data ke list
    except FileNotFoundError:
        print(f"File tidak ditemukan: {file}")
        continue  # Lewati file jika tidak ditemukan

# Gabungkan semua DataFrame dengan menumpuknya
combined_data = pd.concat(data_frames, ignore_index=False)

# Simpan ke file Excel baru
combined_data.to_excel('./testing/2_combined_output.xlsx', index=True)

print("Penggabungan selesai! File disimpan sebagai '2_combined_output.xlsx'.")
