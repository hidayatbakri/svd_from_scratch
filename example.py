"""Contoh penggunaan modul SVD from scratch."""

import numpy as np
from svd_from_scratch import SVD


def main():
    # 1. Definisikan matriks contoh
    matrix = np.array([
        [3.0, 1.0, 2.0],
        [1.0, 3.0, 1.0],
        [4.0, 2.0, 3.0]
    ])

    print("=" * 60)
    print("SVD (Singular Value Decomposition) From Scratch")
    print("=" * 60)
    print("Matriks Asal A:\n", matrix)

    # 2. Inisialisasi dan hitung SVD
    svd = SVD(matrix)
    U, sigma, V = svd.calculate()

    print("\n[Hasil Dekomposisi SVD]")
    print(f"- Singular Values (Sigma):\n  {np.round(sigma, 4)}")
    print(f"- Matriks U (Left Singular Vectors):\n{np.round(U, 4)}")
    print(f"- Matriks V (Right Singular Vectors):\n{np.round(V, 4)}")

    # 3. Truncated SVD / Low-Rank Approximation
    print("\n" + "=" * 60)
    print("Low-Rank Approximation & Frobenius Error")
    print("=" * 60)
    for k in range(1, len(sigma) + 1):
        approx = svd.truncate(k=k)
        error = svd.reconstruction_error(k=k)
        print(f"\n--- Rank-{k} Approximation ---")
        print(np.round(approx, 4))
        print(f"Reconstruction Error (Frobenius Norm): {error:.6f}")


if __name__ == "__main__":
    main()
