import streamlit as st

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pickle

from sklearn import metrics

st.set_page_config(
    page_title="Projectkk"
)
st.title('Klasifikasi Kelayakan Calon Pendonor Darah Menggunakan Metode Naive Bayes')
tab1, tab2, tab3= st.tabs(["Dataset", "Prepocessing","Implementation"])

with tab1:
    
    st.write('Studi Kasus PMI Kabupaten Bangkalan')
    st.write('Darah adalah cairan yang terdapat pada semua makhluk hidup (kecuali tumbuhan) tingkat tinggi yang berfungsi mengirimkan zat – zat dan oksigen yang dibutuhkan oleh jaringan tubuh, mengangkut bahan – bahan kimia hasil metabolisme dan juga sebagai pertahanan tubuh terhadap virus atau bakteri [1] (Desmawati, 2013). ')
    
    st.write('Dalam tubuh orang dewasa, kira – kira 4 sampai 5 liter darah yang beredar terus - menerus melalui jaringan yang rumit mulai dari pembuluh darah, didorong oleh kontraksi kuat dari detak jantung. Setelah darah bergerak menjauh dari paru – paru dan jantung, melewati arteri besar dan mengalir ke jaringan yang sempit dan lebih kompleks dari pembuluh – pembuluh kecil, darah berinteraksi dengan sel – sel individual dari jaringan. Pada tingkat ini, fungsi utamanya adalah untuk memberi makan sel – sel tersebut, memberi mereka nutrisi, termasuk oksigen yang merupakan unsur paling dasar yang diperlukan untuk keberlangsungan hidup manusia. Dalam pertukaran nutrisi bermanfaat ini, darah menggambil dan membawa pergi limbah seluler seperti karbon dioksida yang pada akhirnya akan dikeluarkan dari tubuh ketika darah mengalir kembali ke paru – paru. ')

    df = pd.read_csv("https://raw.githubusercontent.com/dewialqurani/project_kk/main/kelompok8.csv")
    st.write("Dataset Donor Darah : ")
    st.write(df)

    st.write("Penjelasan Nama - Nama Kolom : ")
    st.write("""
    <ol>
    <li>Tempat lahir :Tempat seseorang dilahirkan.</li>
    <li>Tanggal lahir :Identitas kapan dilahirkan ke dunia setelah berada di kandungan.</li>
    <li>Umur : Umur atau usia pada manusia adalah waktu yang terlewat sejak kelahiran.</li>
    <li>Golongan Darah :Golongan darah adalah ilmu pengklasifikasian darah dari suatu kelompok berdasarkan ada atau tidak adanya zat antigen warisan pada permukaan membran sel darah merah. </li>
    <li>Jenis Kelamin :jenis kelamin adalah perbedaan bentuk, sifat, dan fungsi biologis antara laki-laki dan perempuan yang menentukan perbedaan peran mereka dalam menyelenggarakan upaya eneruskan garis keturunan.</li>
    <li>HB :Hb adalah protein yang ada di dalam sel darah merah. Protein inilah yang membuat darah berwarna merah.</li>
    <li>BB (kg) :berat badan tubuh yang memiliki proporsi seimbang dengan tinggi badan.</li>
    <li>Tensi :Tensi normal atau tekanan darah normal adalah ukuran ideal dari kekuatan yang digunakan jantung untuk melakukan fungsinya yaitu memompa darah ke seluruh tubuh.</li>
    <li>Status : status kelayakan donor Darah yang diklasifikasikan ke dalam kelas “BOLEH DONOR” dan “TIDAK BOLEH DONOR” </li>
    </ol>
    """,unsafe_allow_html=True)

with tab2:
    st.write("Data preprocessing adalah proses yang mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini penting dilakukan karena data mentah sering kali tidak memiliki format yang teratur. Selain itu, data mining juga tidak dapat memproses data mentah, sehingga proses ini sangat penting dilakukan untuk mempermudah proses berikutnya, yakni analisis data.")
    st.write("Data preprocessing adalah proses yang penting dilakukan guna mempermudah proses analisis data. Proses ini dapat menyeleksi data dari berbagai sumber dan menyeragamkan formatnya ke dalam satu set data.")
    
    scaler = st.radio(
    "Pilih Metode Normalisasi Data : ",
    ('Tanpa Scaler', 'MinMax Scaler'))
    if scaler == 'Tanpa Scaler':
        st.write("Dataset Tanpa Preprocessing : ")
        df_new=df
    elif scaler == 'MinMax Scaler':
        st.write("Dataset setelah Preprocessing dengan MinMax Scaler: ")
        df=df.drop(['NO','TEMPAT LAHIR','TANGGAL LAHIR'], axis=1)

        from sklearn.preprocessing import LabelEncoder
        labelencoder = LabelEncoder()
        df['GOLONGAN DARAH'] = labelencoder.fit_transform(df['GOLONGAN DARAH'])
        df['JENIS KELAMIN'] = labelencoder.fit_transform(df['JENIS KELAMIN'])
        df['STATUS'] = labelencoder.fit_transform(df['STATUS'])

        scaler = MinMaxScaler()
        df_for_scaler = pd.DataFrame(df, columns = ['UMUR','HB','BB (kg)', 'TENSI', 'TENSI 2'])
        df_for_scaler = scaler.fit_transform(df_for_scaler)
        df_for_scaler = pd.DataFrame(df_for_scaler,columns = ['UMUR','HB','BB (kg)', 'TENSI', 'TENSI 2'])
        df_drop_column_for_minmaxscaler=df.drop(['UMUR','HB','BB (kg)', 'TENSI', 'TENSI 2'], axis=1)
        df_new = pd.concat([df_for_scaler,df_drop_column_for_minmaxscaler], axis=1)
    st.write(df_new)

with tab3:
    st.write("""
    <h5>Implementation Naive Bayes</h5>
    <br>
    """, unsafe_allow_html=True)


    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    df['GOLONGAN DARAH'] = labelencoder.fit_transform(df['GOLONGAN DARAH'])
    df['JENIS KELAMIN'] = labelencoder.fit_transform(df['JENIS KELAMIN'])
    df['STATUS'] = labelencoder.fit_transform(df['STATUS'])

    scaler = MinMaxScaler()

    df_for_scaler = pd.DataFrame(df, columns = ['UMUR','HB','BB (kg)', 'TENSI', 'TENSI 2'])
    df_for_scaler = scaler.fit_transform(df_for_scaler)
    df_for_scaler = pd.DataFrame(df_for_scaler,columns = ['UMUR','HB','BB (kg)', 'TENSI', 'TENSI 2'])
    df_drop_column_for_minmaxscaler=df.drop(['UMUR','HB','BB (kg)', 'TENSI', 'TENSI 2'], axis=1)
    df_new = pd.concat([df_for_scaler,df_drop_column_for_minmaxscaler], axis=1)

    X=df_new.iloc[:,0:7].values
    y=df_new.iloc[:,7].values
    st.write('Jumlah baris dan kolom :', X.shape)
    st.write('Jumlah kelas : ', len(np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    st.write("Data Training :", X_train.shape)
    st.write("Data Testing :", X_test.shape)

    col1,col2 = st.columns([2,2])
    with col1:
        a = st.number_input("Umur",0)
        b = st.number_input("Golongan Darah",0)
        c = st.number_input("Jenis Kelamin",0)
        d = st.number_input("Tensi 2",0)
    with col2:
        e = st.number_input("Hemoglobin",0)
        f = st.number_input("Berat Badan",0)
        g = st.number_input("Tensi",0)
        
    submit = st.button('Prediksi')
    if submit:
        model = GaussianNB()

        model.fit(X_train, y_train)
        Y_pred = model.predict(X_test) 
        score=metrics.accuracy_score(y_test,Y_pred)
        
        X_new = np.array([[a,b,c,d,e,f,g]])
        predict = model.predict(X_new)
        if predict == 0 :
            st.write("""# Pasien Layak Donor Darah""")
        else : 
            st.write("""# Pasien Tidak Layak Donor Darah""")
        st.write(f"akurasi : {score}")
        