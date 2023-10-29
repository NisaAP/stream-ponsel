import pickle
import streamlit as st
import sklearn

#baca model
model = pickle.load(open('estimasi_ponsel.sav','rb'))
#Judul
st.title('Estimasi Harga Ponsel')

#bagi kolom
col1, col2 = st.columns(2)

with col1:
    resoloution = st.number_input('input ukuran resolusi ponsel')
    ppi = st.number_input('input ppi ')
    cpucore = st.number_input('input CPU CORE ')
    cpufreq = st.number_input('input frekuensi cpu')
    internalmem = st.number_input('input kapasitas memori internal')
with col2:
    ram = st.number_input('input RAM')
    RearCam = st.number_input('input kamera belakang')
    Front_Cam = st.number_input('input kamera depan')
    battery = st.number_input('input kapasitas battery')
    thickness= st.number_input('input ketebalan ponsel')
    
predict =''

if st.button('Estimasi Harga '):
    predict = model.predict(
        [[resoloution,ppi,cpucore,cpufreq,internalmem,ram,RearCam,Front_Cam,battery,thickness]]
    )
    st.write('Estimasi Harga Ponsel :',predict)
