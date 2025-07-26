# Auto-generated from PDM.ipynb

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from ast import literal_eval
from collections import Counter


from google.colab import files
uploaded = files.upload()


df = pd.read_csv('BRImoPreProses.csv')

print(df.info())
print(df.head())

sentiment_counts = df['polarity'].value_counts()
print("\nDistribusi Sentimen:")
print(sentiment_counts)

plt.figure(figsize=(6,6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribusi Sentimen (Pie Chart)')
plt.show()

plt.figure(figsize=(8,6))
sentiment_counts.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Distribusi Sentimen (Bar Chart)')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.show()

df['content_tokens_stemmed'] = df['content_tokens_stemmed'].apply(literal_eval)

#pisahkan data positive dan negative
positive_words = df[df['polarity'] == 'positive']['content_tokens_stemmed']
negative_words = df[df['polarity'] == 'negative']['content_tokens_stemmed']

# Gabungkan semua token jadi 1 string
positive_text = ' '.join([' '.join(tokens) for tokens in positive_words])
negative_text = ' '.join([' '.join(tokens) for tokens in negative_words])



custom_stopwords = ['aplikasi', 'brimo', 'bank', 'bri', 'pakai', 'nya', 'aja', 'ga']


custom_stopwords = [
    'aplikasi', 'brimo', 'bank', 'bri',
    'nya', 'pakai', 'pake', 'ga', 'aja',
    'nya', 'nyaa', 'nyaan', 'nyaaa',
    'itu', 'bgt', 'bgtkatanya', 'katanya',
    'dll'
]

stopwords = set(STOPWORDS)
stopwords.update(custom_stopwords)



wc_pos = WordCloud(
    width=800, height=400,
    background_color='white',
    stopwords=stopwords
).generate(positive_text)

wc_neg = WordCloud(
    width=800, height=400,
    background_color='white',
    stopwords=stopwords
).generate(negative_text)

plt.figure(figsize=(16,8))

# Positif
plt.subplot(1,2,1)
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Wordcloud Sentimen Positif (Bersih)')

# Negatif
plt.subplot(1,2,2)
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Wordcloud Sentimen Negatif (Bersih)')

plt.show()

print("\nPOSITIVE TEXT (500 karakter pertama):")
print(positive_text[:500])

print("\nNEGATIVE TEXT (500 karakter pertama):")
print(negative_text[:500])


all_tokens = []
for tokens in df['content_tokens_stemmed']:
    all_tokens.extend(tokens)


word_counts = Counter(all_tokens)

print("\n20 Kata Paling Sering Muncul:")
for word, count in word_counts.most_common(20):
    print(f"{word}: {count}")

target_words = ['mudah', 'cepat', 'error']
print("\nFrekuensi Kata Spesifik:")
for word in target_words:
    print(f"{word}: {word_counts[word]}")
