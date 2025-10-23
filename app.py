import streamlit as st
import re
import joblib
import string
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image

# NLTK 
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


for resource in ["stopwords", "punkt"]:
    try:
        nltk.data.find(f"corpora/{resource}") if resource == "stopwords" else nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

from nltk.corpus import stopwords
nltk_stopwords = set(stopwords.words('indonesian'))

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
st.markdown("""
    <style>
    .scroll-box {
        max-height: 250px;  /* tinggi area scroll */
        overflow-y: scroll;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .scroll-box li {
        margin-bottom: 8px;
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("## 📊 Analisis Tren Judul Penelitian")


#  Preprocessing 
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
base_stopwords = set(stop_factory.get_stop_words())
nltk_stopwords = set(stopwords.words('indonesian'))

# membaca custom stopword
custom_stopwords = {
    "studi","pengaruh","berbasis","penerapan","meningkatkan","pada","untuk","dan","di","yang",
    "pontianak","of","kalimantan","barat","kubu","raya","kota","kabupaten","pengembangan","the","in","indonesia",
    "and","on","peningkatan","kemampuan","abad","abu","acacia","aceh","abdi","ability","accounting","dalam",
    "metode","sistem","model","penggunaan","perancangan","pendekatan","clc","pemanfaatan","pelaksanaan","rancangan",
    "umkm","mentari","fakultas","teknik","kuburaya","laporan","bnnk","p4gn","pgn","mempawah","program","peranan",
    "kawasan","pesisir","lintang","utaea","selatan","solusi","nomor","tahun","kasus","mahasiswa","pendidikan",
    "matmatika","fkip","untan","isbi","singkawang","desa","entikong","kecamatan","pola","malaysia","dampak",
    "keuskupan","agung","paket","ar","ukm","ilmio","gelato","iais","sambas","institut","agama","islam","sultan",
    "muhammad","syafiuddin","4c","c","negeri","kampung","nelayan","sungai","kakap","respon","mea","pb","al-husna",
    "al","husna","ulrich","kelurahab","kedabang","sintang","kayong","utara","bengkayang","samarinda","jalan","imam",
    "bonjol","adi","sucipto","paloh","rth","obia","jakarta","islamic","index","bukit","semuja","taman","nasional",
    "danau","sentarum","daerah","universitas","tanjungpura","das","labang","kebiyau","elais","oleifera","dalbergia",
    "palviflora","roxb","karim","muara","kelal","sandai","ketapang","matapelajaran","sahan","seluas","itb","landak",
    "pt","persepsi","karyawan","mar","crm","provinsi","etr","rirang","jati","sekadau","zea","mays","l","shorea",
    "pinanga","solanum","lycopersicum","durio","sp","ws","kapuas","metharizium","spp","oryctes","rhinocheros",
    "itik","parit","keladi","cutcuma","domestica","val","asia","tenggara","liut","hg","sanggau","nanga","pinoh",
    "palangka","palangkaraya","kb","sekip","lama","tengah","caping","banjarmasin","rhinoplax","vigil","pakintan",
    "nyarumkop","timur","pinyuh","kaskade","kepulau","jawa","pulau","aruk","bumu","lestari","hulu","kph","jongkong",
    "puu","xxi","mk","malilayan","teriak","bk3s","bks","kalbar","perayaan","cap","go","meh","pasir","pbl","hilir",
    "crassicarpa","a","cunn","ex","benth","bumdes","kab","k","k3","no","bersama","sibu","sarawak","bris","kuala",
    "dua","rad","ndc","lol","sejiram","tebas","panjang","simpang","kasturi","mandor","gc","ms","hplc","aloe","vera",
    "terminalia","catappa","nlcs","ambawang","melawi","onchidium","thyphae","rattus","norvegicus","jagoi",
    "calophyllum","soulattri","burm","f","mdr","smpn","rdud","syarif","muhamad","alkadrie","beting","tambelan",
    "kamboja","setapuk","besar","tinospora","crispa","raya","sentagau","jaya","bakau","kecil","ecd","angkatan",
    "bank","btn","cabang","sekitarnya","elte","e","fomo","puteria","compechiana","oss","rba","s","ap","pra","pon",
    "xxl","sumut","pgsd","iot","wsn","garcinia","parvifolia","xi","banyuke","ox","bow","citrus","nobilis","var",
    "microcarpa","samapi","blume","efl","capsicum","annum","chinensis","pouterua","pangkep","sulawesi",
    "amorphophallus","oncophyllus","ai","r","b","nos","need","analysis","kol","cbe","bsps", "ada", "adat", "acid",
    "addie","adsorpsi","adsorben", "actinomycetes", "acorus", "adopsi", "adaptasi", "administrasi","additive","weighting","isolasi",
    "lahan", "aedes", "aegypti", "adaptif", "agen", "adoption", "acceptance",
    "iv","terampil","kuat","video", "siswa","melayu", "inggris", "mandarin", "staphylococcus", "coli", "co",
    "implementasi", "based", "ipa", "ipas", "matematika", "hasil", "ajar", "arab", "saudi", "biologi", "udang"
    "batu", "lingkung", "dayak","smp", "sma", "mts", "kepala", "dasar", "magang", "kosong", "cod", "ispo", "mata", "pelajaran", 
    "kuliah", "efek", "aktivitas", "batu", "ampar", "dayak", "ahe", "alang", "suku", "adiktif", "suku", "akasia", "penerapan", "pengaruh",
    "imperta", "aditif", "albicans", "vitro", "akar", "daun", "udang", "stick", "akurasi",
     "allium", "sativum", "evolusi", "alat", "mesin", "aksi", "akibat",
    "akuntansi", "akuntabilitas", "akademik", "aktif", "aktifitas", "aktifitas", "batu", "ampar",
     "kajian", 
    "data", "hasil", "proses", "pengaruh", "faktor", "penerapan", "ampas", "sagu", "anak", "air", "alam", "algoritma", "analisis", "animosity", "angulata", "alpha"
}


# Gabungkan semua stopword
all_stopwords = base_stopwords.union(nltk_stopwords).union(custom_stopwords)

def cleningText(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    # hapus angka romawi (contoh: i, ii, iii, iv, v, vi, vii, viii, ix, x, xi, xii, dst.)
    text = re.sub(r'\b(?:x{0,3}(?:ix|iv|v?i{0,3}))\b', ' ', text)
    # hapus angka & simbol, sisakan huruf spasi
    text = re.sub(r'[^a-z\s]', ' ', text)
    # menghapus semua tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # menghapus karakter spasi dari kiri dan kanan teks
    text = text.strip()
    # tokenisasi
    tokens = word_tokenize(text)
    # hapus stopword
    tokens = [word for word in tokens if word not in all_stopwords ]
    # gabungakan kembali hasil menjadi satu string
    clean_text = ' '.join(tokens)

    return clean_text


@st.cache_resource
def load_model():
    return joblib.load("assets/model/model_clustering_text.pkl")

model = load_model()


# Dataset 
@st.cache_data
def load_data():
    df = pd.read_csv("assets/dataset/dataset_fix.csv")  
    return df


model = load_model()
df = load_data()

# Input judul dari user

judul_input = st.text_input("Masukkan judul penelitian:")


# Mapping antara nomor cluster dan nama tren/topik
cluster_labels = {
    0: "Evaluasi Kinerja Keuangan di Sebuah Perusahaan",
    1: "Evaluasi Implementasi Kurikulum atau Pembelajaran di Sekolah",
    2: "Implementasi Media atau Teknologi Pembelajaran sebagai Strategi Pembelajaran di Sekolah",
    3: "Sumber Daya Energi Terbarukan",
    4: "Machine Learning, NLP",
    5: "Implementasi Media Belajar Interaktif di Sekolah (Augmented Reality, Video Animasi)",
    6: "Pemanfaatan Jenis Tumbuhan sebagai Obat Tradisional (Etnobotani)",
    7: "Pengembangan Aplikasi Sistem Informasi",
    8: "Kajian Budaya Etnis Sosial Masyarakat Perbatasan",
    9: "Pengembangan Bahan Ajar Calon Guru",
    10: "Uji Efektifitas Tanaman sebagai Penghasil Senyawa Antibakteri dan Kajian Konservasi Lahan Gambut"
}

if judul_input:
    # Bersihkan input
    cleaned_input = cleningText(judul_input)

    # Prediksi cluster menggunakan pipeline yang sudah dilatih
    cluster_pred = model.predict([cleaned_input])[0]
    tren_label = cluster_labels.get(int(cluster_pred), f"Tren #{cluster_pred}")

    st.markdown(
    f"<h5>Judul kamu termasuk dalam <span style='color:#2a9d8f;'>Tren {tren_label}</span></h5>",
    unsafe_allow_html=True)
    # Tampilkan hasil
    # st.markdown(
    #    f"<h5>Judul kamu termasuk dalam Tren #{cluster_pred}</h5>",
    #    unsafe_allow_html=True
    # )
    
    if "cluster" not in df.columns:
        st.warning("Dataset kamu belum punya kolom 'cluster', sehingga tidak bisa menampilkan judul serupa.")
    else:
        st.markdown("### 🔎 Judul Serupa dalam Tren yang Sama:")
        similar_titles = df[df["cluster"] == cluster_pred]["judul_indo"].values.tolist()

        # cluster_pred_value = cluster_pred[0]
        # similar_titles = df[(df["cluster"] == cluster_pred_value) & (df["judul_indo"] != judul_input)]["judul_indo"].values.tolist()

        if len(similar_titles) > 0:
         list_items = "".join([f"<li>{t}</li>" for t in similar_titles])
         st.markdown(f"<div class='scroll-box'><ol>{list_items}</ol></div>", unsafe_allow_html=True)
        else:
         st.write("Tidak ada judul lain dalam tren ini.")

        # ☁️ Word Cloud Tren Ini
        st.markdown("### ☁️ Word Cloud Tren Ini:")

        # Pastikan tipe data sama antara cluster_pred dan kolom cluster
        df["cluster"] = df["cluster"].astype(str)
        cluster_pred = str(cluster_pred)

        # Ambil teks dari cluster yang diprediksi
        cluster_data = df[df["cluster"] == cluster_pred]

        if not cluster_data.empty:
            cluster_text = " ".join(cluster_data["clean_text"].dropna().astype(str))
            
            if cluster_text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(cluster_text)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning("Tidak ada teks valid dalam cluster ini untuk dibuat WordCloud.")
        else:
            st.warning(f"Tidak ditemukan data untuk cluster {cluster_pred}. Coba input judul lain.")

        count_in_cluster = len(similar_titles)
        total_titles = len(df)
        st.info(f"📈 Jumlah penelitian dalam tren ini: **{count_in_cluster} dari {total_titles} total judul.**")    