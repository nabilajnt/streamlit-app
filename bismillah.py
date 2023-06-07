import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
np.bool = bool

# Judul Tugas dan Nama Anggota
st.sidebar.title('Tugas Perancangan Aplikasi Sains Data Kelompok 5')
st.sidebar.write('Anggota:')
st.sidebar.write('1. Aliya Millati Risphi (1305210064)')
st.sidebar.write('2. Nabila Janatri Iswibowo (1305210082)')
st.sidebar.write('3. Chintya Annisah Solin (1305213025)')

st.title('Golongan Darah Predictor')

st.write("""
Masukkan atribut berikut untuk memprediksi golongan darah:
""")

# Mendefinisikan input pengguna
kode_provinsi = st.selectbox("Kode Provinsi", [32])
jumlah_penduduk = st.number_input("Jumlah Penduduk", min_value=0, max_value=10000000, value=0)
jenis_kelamin = st.selectbox("Jenis Kelamin", ["LAKI-LAKI", "PEREMPUAN"])
tahun = st.selectbox("Tahun", list(range(2013, 2022)))

# Membaca dataset golongan darah dari file Excel
data = pd.read_excel("datagoldar.xlsx")

# Filter data berdasarkan tahun yang dipilih
data_tahun = data[data["tahun"] == tahun]

# Melakukan pemrosesan data
# Mengubah jenis kelamin menjadi nilai numerik
data_tahun["jenis_kelamin"] = data_tahun["jenis_kelamin"].map({"LAKI-LAKI": 0, "PEREMPUAN": 1})

# Memisahkan atribut dan target
X = data_tahun[["kode_provinsi", "jumlah_penduduk", "jenis_kelamin"]]
y = data_tahun["golongan_darah"]

# Membuat model Random Forest
model = RandomForestClassifier()
model.fit(X, y)

encoder = LabelEncoder()
jenis_kelamin_encoded = encoder.fit_transform([jenis_kelamin])

# Memprediksi golongan darah berdasarkan input pengguna
prediksi = model.predict([[kode_provinsi, jumlah_penduduk, jenis_kelamin_encoded[0]]])


# Menampilkan hasil prediksi
st.write(f"Hasil Prediksi Golongan Darah: {prediksi[0]}")

# Visualisasi masing-masing golongan darah berdasarkan tahun
for golongan_darah in data_tahun["golongan_darah"].unique():
    golongan_data = data_tahun[data_tahun["golongan_darah"] == golongan_darah]
    plt.scatter(golongan_data["kode_provinsi"], golongan_data["jumlah_penduduk"], label=golongan_darah)

plt.xlabel("Kode Provinsi")
plt.ylabel("Jumlah Penduduk")
plt.legend()
plt.title(f"Visualisasi Golongan Darah Tahun {tahun}")
st.pyplot(plt)
