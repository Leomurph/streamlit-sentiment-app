import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# === Setup halaman ===
st.set_page_config(page_title="BRImo Sentiment Analyzer", layout="wide")
st.title("BRImo Sentiment Analyzer")
st.write("Menganalisis sentimen ulasan pengguna aplikasi BRImo menggunakan Machine Learning.")

# === Upload file CSV ===
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'ulasan' not in df.columns:
        st.error("File harus memiliki kolom bernama 'ulasan'.")
    else:
        st.success("File berhasil diunggah! Menjalankan analisis...")

        # Load model dan vectorizer
        model = pickle.load(open("sentiment_model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

        # Prediksi sentimen
        X = vectorizer.transform(df['ulasan'].astype(str))
        df['hasil'] = model.predict(X)

        # Visualisasi: Distribusi Sentimen
        st.subheader("Distribusi Sentimen")
        st.bar_chart(df['hasil'].value_counts())

        # Visualisasi: WordCloud
        st.subheader("WordCloud dari Ulasan")
        all_text = " ".join(df["ulasan"].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Tampilkan data
        st.subheader("Hasil Analisis")
        st.dataframe(df[['ulasan', 'hasil']])

        # Unduh hasil analisis
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Unduh CSV Hasil", data=csv, file_name="hasil_analisis.csv", mime="text/csv")
