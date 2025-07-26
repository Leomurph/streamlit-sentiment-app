import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from ast import literal_eval
from collections import Counter

st.set_page_config(page_title="Analisis Sentimen BRImo", layout="wide")
st.title("Analisis Sentimen Pengguna Aplikasi BRImo")

# Upload file CSV
uploaded_file = st.file_uploader("Unggah file CSV (contoh: BRImoPreProses.csv)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Data Ulasan")
    st.dataframe(df.head())

    st.subheader("📊 Distribusi Sentimen")
    sentiment_counts = df['polarity'].value_counts()
    col1, col2 = st.columns(2)

    with col1:
        st.write("Pie Chart")
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
        ax1.axis("equal")
        st.pyplot(fig1)

    with col2:
        st.write("Bar Chart")
        fig2, ax2 = plt.subplots()
        sentiment_counts.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'], ax=ax2)
        ax2.set_title('Distribusi Sentimen')
        ax2.set_xlabel('Sentimen')
        ax2.set_ylabel('Jumlah')
        st.pyplot(fig2)

    st.subheader("☁️ WordCloud Positif & Negatif")
    df['content_tokens_stemmed'] = df['content_tokens_stemmed'].apply(literal_eval)

positive_words = df[df['polarity'] == 'positive']['content_tokens_stemmed']
negative_words = df[df['polarity'] == 'negative']['content_tokens_stemmed']

if 'polarity' in df.columns and 'content_tokens_stemmed' in df.columns:
    # Lanjut proses analisis
    positive_words = df[df['polarity'] == 'positive']['content_tokens_stemmed']
    ...
else:
    st.error("File CSV tidak memiliki kolom 'polarity' dan/atau 'content_tokens_stemmed'. Pastikan file sudah diproses.")

