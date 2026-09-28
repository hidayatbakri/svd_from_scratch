import numpy as np

class SVD:
  def __init__(self, matrix: np.ndarray):
    self.matrix = matrix.astype(float)
    self.U = None
    self.sigma = None
    self.V = None

  def calculate(self):
    MT_M = np.matmul(self.matrix.T, self.matrix)
    eigenvalue, V = np.linalg.eigh(MT_M)
    idx = np.argsort(eigenvalue)[::-1]
    self.sigma = np.sqrt(eigenvalue)[idx]
    self.V = V[:, idx]

    self.U = np.matmul(self.matrix, self.V) / self.sigma.reshape(1, -1)
    return self.U, self.sigma, self.V

  def truncate(self, k: int = 1):

    if self.U is None:
      self.calculate()

    max_possible_rank = len(self.sigma)
    if k > max_possible_rank or k < 1:
      raise ValueError(f"Nilai k harus antara 1 dan {max_possible_rank}")

    U_k = self.U[:, :k]
    sigma_k = self.sigma[:k]
    V_k = self.V[:, :k]

    matrix_k = np.matmul(np.matmul(U_k, np.diag(sigma_k)), V_k.T)
    return matrix_k

  def reconstruction_error(self, k: int = 1) -> float:
    approx = self.truncate(k)
    return float(np.linalg.norm(self.matrix - approx, ord='fro'))

if __name__ == "__main__":
    W = np.array([[3, 1, 2], 
                  [1, 3, 1],
                  [4, 2, 3]])

    svd = SVD(W)
    U, sigma, V = svd.calculate()

    print("=== Full SVD ===")
    print("Sigma:", sigma)
    print("U:\n", np.round(U, 4))
    print("V:\n", np.round(V, 4))

    k = 2
    print(f"\n=== Rank-{k} Approximation ===")
    W_rank1 = svd.truncate(k=k)
    print(np.round(W_rank1, 4))

    print("\nError Rekonstruksi (Frobenius):", round(svd.reconstruction_error(k=k), 4))