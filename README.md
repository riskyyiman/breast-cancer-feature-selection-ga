# Feature Selection Menggunakan Genetic Algorithm pada Wisconsin Breast Cancer Dataset

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

Proyek ini bertujuan untuk mengatasi masalah dimensionalitas fitur pada _Wisconsin Breast Cancer Diagnostic Dataset_ (WBCD). Dengan menerapkan **Genetic Algorithm (GA)** berbasis pendekatan _wrapper_, sistem ini memilih subset fitur yang paling relevan untuk meningkatkan akurasi klasifikasi sekaligus mengurangi kompleksitas komputasi pada sistem _computer-aided diagnosis_ (CAD).

## 📌 Latar Belakang

Dataset medis WBCD memiliki 30 fitur numerik yang dapat menyebabkan _overfitting_ dan beban komputasi tinggi jika digunakan seluruhnya. Penelitian ini menggunakan GA untuk melakukan eksplorasi global dalam mencari kombinasi fitur optimal guna mendapatkan performa model yang lebih baik dibandingkan menggunakan seluruh fitur.

## 🛠️ Metodologi & Parameter

Sistem menggunakan **Support Vector Machine (SVM)** sebagai _classifier_ untuk mengevaluasi setiap subset fitur yang dihasilkan oleh algoritma genetika.

### Parameter Genetic Algorithm:

| Parameter                  | Nilai / Metode      | Justifikasi                                                            |
| :------------------------- | :------------------ | :--------------------------------------------------------------------- |
| **Ukuran Populasi**        | 50                  | Memberikan keragaman individu yang cukup tanpa membebani memori.       |
| **Generasi Maksimum**      | 100                 | Batas iterasi untuk memastikan algoritma mencapai konvergensi optimal. |
| **Probabilitas Crossover** | 0.8 (80%)           | Memfasilitasi pertukaran informasi genetik untuk eksplorasi solusi.    |
| **Probabilitas Mutasi**    | 0.05 (5%)           | Menjaga variasi genetik agar tidak terjebak pada _local optima_.       |
| **Metode Seleksi**         | Tournament (Size=3) | Menjaga tekanan seleksi agar individu terbaik tetap terpilih.          |
| **Elitisme**               | Ya (Top 2)          | Menjamin solusi terbaik tidak hilang selama proses evolusi.            |

## 📊 Dataset

Dataset yang digunakan berasal dari **UCI Machine Learning Repository**:

- **Sampel**: 569 data pasien.
- **Fitur**: 30 fitur numerik (radius, tekstur, perimeter, area, dll).
- **Kelas**: _Benign_ (Jinak) dan _Malignant_ (Ganas).

## 🚀 Instalasi & Penggunaan

1. Clone repositori ini.
2. Instal pustaka yang diperlukan:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```
3. Jalankan Program
   python app.py

📈 Hasil Eksperimen
Reduksi Fitur : Berhasil mengurangi dimensi data dari 30 fitur menjadi sekitar 16 fitur utama.
Akurasi : Mengalami peningkatan stabilitas akurasi hingga mencapai ~98% menggunakan 10-Fold Cross Validation.
Konvergensi : Algoritma mencapai titik optimal secara stabil pada generasi ke-60 hingga ke-80.


Kontributor: Risky Iman Lael Prasetio
