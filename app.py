import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Klasifikasi Risiko Kanker Paru-Paru dengan Naive Bayes")

# Upload file
uploaded_file = st.file_uploader("Upload Dataset (.csv)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Dataset")
    st.dataframe(df)

    # Encode data
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df_encoded[col] = le.fit_transform(df[col])

    if 'Hasil' not in df.columns:
        st.error("Kolom 'Hasil' (target) tidak ditemukan.")
    else:
        X = df_encoded.drop(columns=['Hasil', 'No']) if 'No' in df.columns else df_encoded.drop(columns=['Hasil'])
        y = df_encoded['Hasil']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("Akurasi Model")
        st.write(f"{acc * 100:.2f}%")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Tidak", "Ya"], yticklabels=["Tidak", "Ya"])
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        st.pyplot(fig)
else:
    st.info("Silakan upload dataset CSV terlebih dahulu.")
