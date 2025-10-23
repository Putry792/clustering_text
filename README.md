# 🧠 Text Clustering App

Proyek ini merupakan aplikasi **klasterisasi teks** berbasis Python dan Streamlit yang digunakan untuk mengelompokkan judul penelitian menggunakan teknik **TF-IDF**, **K-Means**, dan visualisasi seperti **WordCloud**.  
Proyek ini dijalankan menggunakan **virtual environment (venv)** agar dependensi tetap terisolasi.

---

## 📂 Struktur Folder
```
clustering_text/
│
├── app.py # File utama Streamlit
├── model/ # Folder model (jika ada)
├── data/ # Dataset (opsional)
├── requirements.txt # Daftar dependensi
└── README.md
```
## ⚙️ Cara Menjalankan Proyek

### 1️⃣ Clone Repository
```bash

git clone https://github.com/username/text-clustering.git
cd text-clustering
```
### 2️⃣ Buat Virtual Environment
```bash

python -m venv venv

```
### 3️⃣ Aktifkan Virtual Environment
Windows
```bash
venv\Scripts\activate
```
Mac/Linux
```bash
venv\Scripts\activate
```
### 4️⃣ Instal Dependensi
```bash
pip install -r requirements.txt
```

### Jalankan Aplikasi Streamlit
```bash
streamlit run app.py

```
Aplikasi akan berjalan di browser di alamat:
```bash
http://localhost:8501


```
