from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import numpy as np

from sklearn import datasets #panggil lib scikitlearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# bikin fungsi pilih_klasifikasi
def pilih_klasifikasi(nama_algoritma, params):
    algo = None
    if nama_algoritma == 'KNN':
        algo = KNeighborsClassifier(n_neighbors=params['K'])
    elif nama_algoritma == 'SVM':
        algo = SVC(C=params['C'])
    else:
        algo = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
    return algo

# tampilkan hasil algoritma sesuai parameter
algo = pilih_klasifikasi(algoritma, params)

### PROSES KLASIFIKASI ###
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234) 

algo.fit(x_train, y_train)
y_pred = algo.predict(x_test) #panggil fungsi prediksi

acc = accuracy_score(y_test, y_pred)

st.write(f'Algoritma = {algoritma}')
st.write(f'Akurasi =', acc)