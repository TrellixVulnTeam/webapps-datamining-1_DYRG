import streamlit as st
import numpy as np

from sklearn import datasets #panggil lib scikitlearn

st.title('Aplikasi Web Data Mining')
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

# buat function untuk tambah parameter
def tambah_parameter(nama_algortima):
    params = dict()
    if nama_algortima == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif nama_algortima == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

# panggil fungsi params berdasarkan algoritma yg dipilih
params = tambah_parameter(algoritma)
