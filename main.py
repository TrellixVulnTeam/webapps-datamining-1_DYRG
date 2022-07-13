import streamlit as st
import numpy as np

st.title('Aplikasi Web Datamining')
st.write("""
# Algoritma KNN, SVM dan Random Forest
Mana yang terbaik??
""")

nama_dataset = st.sidebar.selectbox(
    'Pilih Dataset',
    ('Bunga IRIS','Kanker Payudara','Digit Angka')
)

st.write(f"## Dataset {nama_dataset}")