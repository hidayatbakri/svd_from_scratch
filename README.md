# 📐 svd_from_scratch: Singular Value Decomposition from Scratch

Implementasi murni algoritma **Singular Value Decomposition (SVD)** menggunakan Python dan NumPy tanpa menggunakan fungsi bawaan `numpy.linalg.svd`. Proyek `svd_from_scratch` dibuat sebagai modul pembelajaran mandiri (*study project*) dalam kurikulum **AI Engineering** untuk memperdalam pemahaman aljabar linier numerik, kompresi matriks, dan estimasi reduksi dimensi (*low-rank matrix approximation*).

---

## 📌 Daftar Isi
- [Latar Belakang & Motivasi](#-latar-belakang--motivasi)
- [Fondasi Matematika](#-fondasi-matematika)
  - [1. Persamaan SVD](#1-persamaan-svd)
  - [2. Langkah Algoritma](#2-langkah-algoritma)
  - [3. Truncated SVD (Low-Rank Approximation)](#3-truncated-svd-low-rank-approximation)
  - [4. Evaluasi Error (Frobenius Norm)](#4-evaluasi-error-frobenius-norm)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi](#-instalasi)
- [Panduan Penggunaan](#-panduan-penggunaan)
  - [Quick Start](#quick-start)
  - [API Reference](#api-reference)
- [Pengujian Unit (Unit Testing)](#-pengujian-unit-unit-testing)
- [Relevansi dalam AI & Machine Learning](#-relevansi-dalam-ai--machine-learning)
- [Lisensi](#-lisensi)

---

## 🎯 Latar Belakang & Motivasi

Sebagai seorang **AI Engineer**, memahami konsep dasar dekomposisi matriks sangat esensial karena SVD merupakan fondasi di balik:
- **Principal Component Analysis (PCA)** untuk reduksi dimensionalitas.
- **Latent Semantic Analysis (LSA)** pada Natural Language Processing.
- **Sistem Rekomendasi** (Collaborative Filtering / Matrix Factorization).
- **Kompresi Data / Gambar** dan konsep modern seperti **LoRA (Low-Rank Adaptation)** pada LLM.

Proyek `svd_from_scratch` bertujuan mengimplementasikan proses komputasi SVD dari *first principles* menggunakan *eigendecomposition* pada matriks kovarians/gramian.

---

## 🔬 Fondasi Matematika

### 1. Persamaan SVD
Setiap matriks riil $A \in \mathbb{R}^{m \times n}$ dapat didekomposisi menjadi tiga matriks:

$$A = U \Sigma V^T$$

Di mana:
- $U \in \mathbb{R}^{m \times m}$: Matriks ortogonal berisi *Left Singular Vectors* (vektor eigen dari $A A^T$).
- $\Sigma \in \mathbb{R}^{m \times n}$: Matriks diagonal berisi *Singular Values* ($\sigma_1 \ge \sigma_2 \ge \dots \ge 0$).
- $V \in \mathbb{R}^{n \times n}$: Matriks ortogonal berisi *Right Singular Vectors* (vektor eigen dari $A^T A$).

---

### 2. Langkah Algoritma
Implementasi pada proyek ini (`svd_from_scratch/core.py`) menggunakan pendekatan *Eigendecomposition*:

1. **Hitung Matriks Simetris $A^T A$:**
   $$M = A^T A$$
2. **Hitung Eigenvalues ($\lambda_i$) dan Eigenvectors ($V$):**
   Menggunakan dekomposisi eigen pada matriks Hermitian/simetris:
   $$M v_i = \lambda_i v_i$$
3. **Urutkan Nilai Eigen secara Descending:**
   $$\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0$$
4. **Hitung Nilai Singular ($\Sigma$):**
   $$\sigma_i = \sqrt{\lambda_i}$$
5. **Hitung Matriks $U$:**
   $$u_i = \frac{A v_i}{\sigma_i} \implies U = A V \Sigma^{-1}$$

---

### 3. Truncated SVD (Low-Rank Approximation)
Berdasarkan teorema **Eckart–Young–Mirsky**, aproksimasi rank-$k$ terbaik untuk matriks $A$ diperoleh dengan mengambil $k$ nilai singular terbesar:

$$A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

---

### 4. Evaluasi Error (Frobenius Norm)
Untuk mengukur seberapa dekat aproksimasi $A_k$ dengan matriks asli $A$:

$$\|A - A_k\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n (A_{ij} - (A_k)_{ij})^2} = \sqrt{\sum_{i=k+1}^{\min(m,n)} \sigma_i^2}$$

---

## 📁 Struktur Proyek

```text
svd_from_scratch/
├── svd_from_scratch/
│   ├── __init__.py         # Package entrypoint (mengekspor SVD)
│   └── core.py             # Implementasi inti kelas SVD from scratch
├── tests/
│   └── test_svd.py         # Unit tests (Pytest)
├── example.py              # Skrip demonstrasi & kalkulasi
├── requirements.txt        # Dependensi proyek
├── pytest.ini              # Konfigurasi pengujian pytest
├── .gitignore              # Konfigurasi ignore file git
└── README.md               # Dokumentasi proyek
```

---

## ⚙️ Instalasi

Pastikan Python (versi $\ge 3.8$) telah terpasang.

1. **Clone repository ini / buka direktori proyek:**
   ```bash
   cd svd_from_scratch
   ```

2. **(Opsional) Buat dan aktifkan virtual environment:**
   ```bash
   python -m venv venv
   # Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Panduan Penggunaan

### Quick Start
Jalankan skrip demonstrasi:
```bash
python example.py
```

### Contoh Kode Python
```python
import numpy as np
from svd_from_scratch import SVD

# 1. Buat matriks input
A = np.array([
    [3.0, 1.0, 2.0],
    [1.0, 3.0, 1.0],
    [4.0, 2.0, 3.0]
])

# 2. Inisialisasi SVD
model = SVD(A)

# 3. Hitung dekomposisi
U, sigma, V = model.calculate()

print("Singular Values:", sigma)
print("Left Singular Vectors (U):\n", U)
print("Right Singular Vectors (V):\n", V)

# 4. Truncated SVD (Aproksimasi Rank-2)
A_rank2 = model.truncate(k=2)
print("Rank-2 Approximation:\n", A_rank2)

# 5. Error Rekonstruksi (Frobenius Norm)
error = model.reconstruction_error(k=2)
print(f"Reconstruction Error: {error:.6f}")
```

---

## 📖 API Reference

### `class SVD(matrix: np.ndarray)`
Kelas representasi Singular Value Decomposition.

| Metode | Parameter | Return Type | Deskripsi |
| :--- | :--- | :--- | :--- |
| `calculate()` | - | `Tuple[np.ndarray, np.ndarray, np.ndarray]` | Menghitung dan mengembalikan tuple `(U, sigma, V)`. |
| `truncate(k: int = 1)` | `k` *(int)*: rank target | `np.ndarray` | Menghasilkan aproksimasi matriks rank-$k$ ($A_k = U_k \Sigma_k V_k^T$). |
| `reconstruction_error(k: int = 1)` | `k` *(int)*: rank target | `float` | Menghitung selisih error Frobenius norm $\|A - A_k\|_F$. |

---

## 🧪 Pengujian Unit (Unit Testing)

Proyek ini dilengkapi dengan unit test berbasis `pytest` untuk memvalidasi:
- Rekonstruksi penuh matriks ($U \Sigma V^T \approx A$).
- Kesesuaian nilai singular dengan `numpy.linalg.svd`.
- Fungsionalitas pemotongan rank ($k$-rank approximation) dan penanganan nilai $k$ invalid.
- Sifat monotonik penurunan error Frobenius seiring meningkatnya nilai rank $k$.

Jalankan pengujian dengan perintah:
```bash
pytest
```

Output:
```text
============================= test session starts =============================
collected 4 items

tests/test_svd.py ....                                                   [100%]

============================== 4 passed in 0.12s ==============================
```

---

## 💡 Relevansi dalam AI & Machine Learning

| Topik AI | Penerapan SVD |
| :--- | :--- |
| **Data Compression** | Mengurangi ukuran penyimpanan matriks fitur atau gambar beresolusi tinggi dengan mempertahankan singular values terbesar. |
| **Dimensionality Reduction** | Merupakan dasar matematika dari PCA (Principal Component Analysis) untuk mereduksi *curse of dimensionality*. |
| **Recommendation Systems** | Mengisi nilai matriks pengguna-item (*sparse matrix*) melalui faktorisasi matriks tingkat rendah (*low-rank matrix completion*). |
| **LoRA (LLM Fine-tuning)** | Konsep dekomposisi matriks bobot $\Delta W = B \cdot A$ terinspirasi dari prinsip aproksimasi low-rank SVD. |

---

## 📄 Lisensi
Proyek ini dilisensikan di bawah [MIT License](LICENSE). Dibuat untuk tujuan edukasi dan eksplorasi konsep matematika AI.
