import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import CategoricalNB

st.set_page_config(page_title="Klasifikasi Obatan", page_icon=":tada:", layout="wide")
def create_page(content, page_title=""):
    st.title(page_title)
    st.write(content)
    
with st.container():
        selected ="Supervise Drug"
        content = """
        \nWeb ini berfungsi untuk mengklasifikasi untuk 
        Jenis Obat yang sesuai dengan pasien bedasarkan Umur, Jenis Kelamin, Kolesterol, Tekanan Darah, dan Rasio natrium dan kalium dalam darah.
        Web Machine Learning ini menggunakan Algoritma Naive Bayes

        Keterangan Data\n
        1. Age = Umur Pasien\n
        2. Sex = Jenis Kelamin Pasien\n
        3. BP = Tekanan Darah\n
        4. Na_to_K = Rasio natrium dan kalium dalam darah\n
        5. Drug = Type Obat

        Anggota Kelompok
        - Zhekkar Harummy Pramesty 
        - Yuma Arganto
        - Dini Dwi Rahayu
        """
        create_page(content, page_title=selected)

        uploaded_file = st.file_uploader("File csv", type=["csv"])
        if uploaded_file:
                df = pd.read_csv(uploaded_file)
                data = df.dropna()
                st.dataframe(data)
                ########################################################################
                st.write("---")
                st.success("Menghitung skewness")
                #menghitung skewness dari kolom Age pada dataframe data
                skewAge = data.Age.skew(axis = 0, skipna = True)
                st.write("Age skewness: ", skewAge)
                #menghitung skewness dari kolom Na_to_K pada dataframe data
                skewNatoK = data.Na_to_K.skew(axis = 0, skipna = True)
                st.write("Na to K skewness: ", skewNatoK)
                #######################################################################
                st.write("---")
                st.success("Distribusi dari kolom Age dalam dataset")
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.distplot(data['Age'], ax=ax)
                st.pyplot(fig)
                #######################################################################
                st.write("---")
                st.success("Distribusi dari kolom Na_to_K dalam dataset")
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.distplot(data['Na_to_K'], ax=ax)
                st.pyplot(fig)
                st.write("---")
                #######################################################################
                # Binning kolom Age
                bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
                category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
                data['Age_binned'] = pd.cut(data['Age'], bins=bin_age, labels=category_age)
                df_drug = data.drop(['Age'], axis = 1)

                 # Binning kolom Na_to_K
                bin_NatoK = [0, 9, 19, 29, 50]
                category_NatoK = ['<10', '10-20', '20-30', '>30']
                data['Na_to_K_binned'] = pd.cut(data['Na_to_K'], bins=bin_NatoK, labels=category_NatoK)
                data = data.drop(['Na_to_K'], axis = 1)
                ######################################################################

                X = data.drop(["Drug"], axis=1)
                y = data["Drug"]

                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

                # Olah dataframe dengan pd.get_dummies()
                X_train = pd.get_dummies(X_train)
                X_test = pd.get_dummies(X_test)

                #st.write("---")
                st.success("Oversampling dengan SMOTE")
                X_train, y_train = SMOTE().fit_resample(X_train, y_train)
                
                # Set theme for seaborn
                sns.set_theme(style="darkgrid")

                # Create the plot
                fig, ax = plt.subplots()
                sns.countplot(y=y_train, data=data, palette="mako_r", ax=ax)
                plt.ylabel('Drug Type')
                plt.xlabel('Total')
                st.pyplot(fig)
                
                ##############################################################################################
                train_button = st.button("Latih model")
                if train_button:
                    st.write("---")
                    st.success("Training the model")
                    # Training the model
                    NBclassifier1 = CategoricalNB()
                    NBclassifier1.fit(X_train, y_train)

                    # Testing the model
                    y_pred = NBclassifier1.predict(X_test)

                    # Menampilkan hasil classification report, confusion matrix, dan accuracy
                    st.write("Classification Report:")
                    st.write(classification_report(y_test, y_pred))
                    st.write("Confusion Matrix:")
                    st.write(confusion_matrix(y_test, y_pred))
                    NBAcc1 = accuracy_score(y_pred,y_test)
                    st.write("Naive Bayes accuracy is: {:.2f}%".format(NBAcc1*100))

                    #######################################################################
                    pred_lr = NBclassifier1.predict(X_test)
                    prediction = pd.DataFrame({'Sex_F': X_test.loc[:,"Sex_F"], 
                                            'Sex_M': X_test.loc[:,"Sex_M"], 
                                            'BP_HIGH': X_test.loc[:,"BP_HIGH"], 
                                            'BP_LOW': X_test.loc[:,"BP_LOW"],
                                            'BP_NORMAL': X_test.loc[:,"BP_NORMAL"],
                                            'Cholesterol_HIGH': X_test.loc[:,"Cholesterol_HIGH"],
                                            'Cholesterol_NORMAL': X_test.loc[:,"Cholesterol_NORMAL"],
                                            'Age_binned_<20s': X_test.loc[:,"Age_binned_<20s"],
                                            'Age_binned_20s': X_test.loc[:,"Age_binned_20s"],
                                            'Age_binned_30s': X_test.loc[:,"Age_binned_30s"],
                                            'Age_binned_40s': X_test.loc[:,"Age_binned_40s"],
                                            'Age_binned_50s': X_test.loc[:,"Age_binned_50s"],
                                            'Age_binned_60s': X_test.loc[:,"Age_binned_60s"],
                                            'Age_binned_>60s': X_test.loc[:,"Age_binned_>60s"],
                                            'Na_to_K_binned_<10': X_test.loc[:,"Na_to_K_binned_<10"],
                                            'Na_to_K_binned_10-20': X_test.loc[:,"Na_to_K_binned_10-20"],
                                            'Na_to_K_binned_20-30': X_test.loc[:,"Na_to_K_binned_20-30"],
                                            'Na_to_K_binned_>30': X_test.loc[:,"Na_to_K_binned_>30"],'DrugType': pred_lr})
                    # Sex
                    prediction['Sex_F'] = prediction['Sex_F'].replace([1, 0],['Female', 'Male'])

                    #BP
                    prediction['BP_HIGH'] = prediction['BP_HIGH'].replace([1, 0],['High',''])
                    prediction['BP_LOW'] = prediction['BP_LOW'].replace([1, 0],['Low', ''])
                    prediction['BP_NORMAL'] = prediction['BP_NORMAL'].replace([1, 0],['Normal', ''])

                    prediction['BP_HIGH'] = np.where((prediction['BP_HIGH'] == ''), prediction['BP_LOW'], prediction['BP_HIGH'])
                    prediction['BP_HIGH'] = np.where((prediction['BP_HIGH'] == ''), prediction['BP_NORMAL'], prediction['BP_HIGH'])

                    #Cholestrol
                    prediction['Cholesterol_HIGH'] = prediction['Cholesterol_HIGH'].replace([1, 0],['High', 'Normal'])

                    #Age_binned
                    prediction['Age_binned_<20s'] = prediction['Age_binned_<20s'].replace([1, 0],['<20s',''])
                    prediction['Age_binned_20s'] = prediction['Age_binned_20s'].replace([1, 0],['20s',''])
                    prediction['Age_binned_30s'] = prediction['Age_binned_30s'].replace([1, 0],['30s',''])
                    prediction['Age_binned_40s'] = prediction['Age_binned_40s'].replace([1, 0],['40s',''])
                    prediction['Age_binned_50s'] = prediction['Age_binned_50s'].replace([1, 0],['50s',''])
                    prediction['Age_binned_60s'] = prediction['Age_binned_60s'].replace([1, 0],['60s',''])
                    prediction['Age_binned_>60s'] = prediction['Age_binned_>60s'].replace([1, 0],['>60s',''])

                    prediction['Age_binned_<20s'] = np.where((prediction['Age_binned_<20s'] == ''), prediction['Age_binned_20s'], prediction['Age_binned_<20s'])
                    prediction['Age_binned_<20s'] = np.where((prediction['Age_binned_<20s'] == ''), prediction['Age_binned_30s'], prediction['Age_binned_<20s'])
                    prediction['Age_binned_<20s'] = np.where((prediction['Age_binned_<20s'] == ''), prediction['Age_binned_40s'], prediction['Age_binned_<20s'])
                    prediction['Age_binned_<20s'] = np.where((prediction['Age_binned_<20s'] == ''), prediction['Age_binned_50s'], prediction['Age_binned_<20s'])
                    prediction['Age_binned_<20s'] = np.where((prediction['Age_binned_<20s'] == ''), prediction['Age_binned_60s'], prediction['Age_binned_<20s'])
                    prediction['Age_binned_<20s'] = np.where((prediction['Age_binned_<20s'] == ''), prediction['Age_binned_>60s'], prediction['Age_binned_<20s'])

                    #Na to K
                    prediction['Na_to_K_binned_<10'] = prediction['Na_to_K_binned_<10'].replace([1, 0],['<10',''])
                    prediction['Na_to_K_binned_10-20'] = prediction['Na_to_K_binned_10-20'].replace([1, 0],['10-20',''])
                    prediction['Na_to_K_binned_20-30'] = prediction['Na_to_K_binned_20-30'].replace([1, 0],['20-30',''])
                    prediction['Na_to_K_binned_>30'] = prediction['Na_to_K_binned_>30'].replace([1, 0],['>30s',''])

                    prediction['Na_to_K_binned_<10'] = np.where((prediction['Na_to_K_binned_<10'] == ''), prediction['Na_to_K_binned_10-20'], prediction['Na_to_K_binned_<10'])
                    prediction['Na_to_K_binned_<10'] = np.where((prediction['Na_to_K_binned_<10'] == ''), prediction['Na_to_K_binned_20-30'], prediction['Na_to_K_binned_<10'])
                    prediction['Na_to_K_binned_<10'] = np.where((prediction['Na_to_K_binned_<10'] == ''), prediction['Na_to_K_binned_>30'], prediction['Na_to_K_binned_<10'])

                    # Drop columns
                    prediction = prediction.drop(['Sex_M', 'BP_LOW', 'BP_NORMAL', 'Cholesterol_NORMAL', 'Age_binned_20s', 'Age_binned_30s',
                                    'Age_binned_40s', 'Age_binned_50s', 'Age_binned_60s', 'Age_binned_>60s',
                                    'Na_to_K_binned_10-20', 'Na_to_K_binned_20-30', 'Na_to_K_binned_>30'], axis = 1)
                    
                    new_name = {'Sex_F': 'Sex', 'BP_HIGH': 'BP', 'Cholesterol_HIGH': 'Cholesterol', 'Age_binned_<20s': 'Age_binned',
                                'Na_to_K_binned_<10': 'Na_to_K_binned'}
                    prediction.rename(columns=new_name, inplace=True)

                    #Hasil Akhir
                    st.success("Hasil Akhir Klasifikasi ")
                    st.dataframe(prediction)