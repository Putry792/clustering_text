import streamlit as st
import re
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from PIL import Image

#  Konfigurasi Halaman 
st.set_page_config(page_title="Prediksi Tren Penelitian", layout="centered")

#  Sidebar 
st.sidebar.title("📘 Tentang Aplikasi")
st.sidebar.write("""
Aplikasi ini digunakan untuk **menganalisis tren penelitian** berdasarkan **judul penelitian** yang kamu masukkan.

🔹 Mengelompokkan judul-judul penelitian ke dalam beberapa tren  
🔹 Menampilkan judul serupa yang termasuk tren yang sama  
🔹 Menampilkan *word cloud* kata dominan tiap tren  

Gunakan aplikasi ini untuk melihat apakah topikmu **sudah populer atau masih jarang diteliti** 💡
""")

st.sidebar.markdown("---")
st.sidebar.caption("""
 👩‍💻 **Dikembangkan oleh:**  
    Prodi D3 Teknologi Informasi  
    Politeknik Negeri Ketapang  

    🧩 **Versi:** 1.0 
                   """)

#  Header Gambar 
# with st.container():
#     image = Image.open("assets/img-1.jpg")
#     st.image(image, use_container_width=True, caption="Temukan tren topik penelitianmu!")

st.markdown("## 📊 Analisis Tren Judul Penelitian")

#  Dummy Dataset 
dummy_titles = [
    "Analisis Sentimen Pengguna Twitter terhadap Kebijakan Pemerintah",
    "Penerapan Machine Learning untuk Prediksi Harga Saham",
    "Perancangan Aplikasi Pembelajaran Berbasis Android",
    "Sistem Rekomendasi Film Menggunakan Collaborative Filtering",
    "Studi Pengaruh Media Sosial terhadap Perilaku Konsumen",
    "Klasifikasi Tanaman Berdasarkan Citra Daun Menggunakan CNN",
    "Perancangan Sistem Informasi Akademik di Sekolah",
    "Deteksi Hoaks pada Berita Online Menggunakan NLP",
    "Prediksi Cuaca dengan Metode Regresi Linear",
    "Analisis Pengaruh E-Commerce terhadap UMKM di Indonesia",
    "Implementasi Deep Learning untuk Deteksi Penyakit Daun",
    "Pengaruh Teknologi Informasi terhadap Efisiensi Bisnis",
    "Sistem Pendeteksi Emosi dari Suara Menggunakan AI",
    "Perancangan Aplikasi Absensi Menggunakan QR Code",
    "Penerapan Data Mining untuk Analisis Pola Belanja Konsumen"
]

#  Preprocessing 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_data = [clean_text(t) for t in dummy_titles]

# TF-IDF dan clustering dummy
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(cleaned_data)
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
clusters = model.labels_

df = pd.DataFrame({
    "judul": dummy_titles,
    "cluster": clusters
})

#  Input Judul 
# st.markdown("Masukkan Judul Penelitian Kamu")
judul_input = st.text_input("Masukkan judul penelitian:")

if judul_input:
    cleaned_input = clean_text(judul_input)
    input_vec = vectorizer.transform([cleaned_input])
    cluster_pred = model.predict(input_vec)[0]

    st.markdown(
       f"<h5>Judul kamu termasuk dalam Tren #{cluster_pred}</h5>",
       unsafe_allow_html=True
    )

    #  Daftar Judul Serupa 
    st.markdown("### 🔎 Judul Serupa dalam Tren yang Sama:")
    similar_titles = df[df["cluster"] == cluster_pred]["judul"].values.tolist()

    for i, t in enumerate(similar_titles, 1):
        st.markdown(f"{i}. {t}")

    #  Word Cloud 
    st.markdown("### ☁️ Word Cloud Tren Ini:")
    cluster_text = " ".join(df[df["cluster"] == cluster_pred]["judul"])

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(cluster_text)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    #  Jumlah Penelitian dalam Tren 
    count_in_cluster = len(similar_titles)
    total_titles = len(df)
    st.info(f"📈 Jumlah penelitian dalam tren ini: **{count_in_cluster} dari {total_titles} total judul.**")

else:
    st.info("💡 Masukkan judul penelitian untuk melihat tren dan topik yang serupa.")