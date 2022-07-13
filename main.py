import streamlit as st
import numpy as np

from sklearn import datasets #panggil lib scikitlearn

st.title('Aplikasi Web Datamining')
st.write("""
# Algoritma KNN, SVM dan Random Forest
Mana yang terbaik??
""")

# sidebar pilih dataset
nama_dataset = st.sidebar.selectbox(
    'Pilih Dataset',
    ('Bunga IRIS','Kanker Payudara','Digit Angka')
)

# tampilkan dataset sesuai kondisi di sidebar
st.write(f"## Dataset {nama_dataset}")

# sidebar pilih algoritma
algoritma = st.sidebar.selectbox(
    'Pilih Algoritma',
    ('KNN','SVM','Random Forest')
)

# buat function
def pilih_dataset(nama):
    data = None
    if nama == 'Bunga IRIS':
        data = datasets.load_iris()
    elif nama == 'Kanker Payudara':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_digits()
    x = data.data
    y = data.target
    return x,y

# kondisi yang ditampilkan berdasarkan dataset
x, y = pilih_dataset(nama_dataset)
st.write('Jumlah Baris dan kolom : ', x.shape)
st.write('Jumlah kelas : ', len(np.unique(y)))