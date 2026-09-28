import numpy as np
import pytest
from svd_from_scratch import SVD


def test_svd_reconstruction():
    """Menguji bahwa rekontruksi rank penuh mendekati matriks asli."""
    A = np.array([
        [3.0, 1.0, 2.0],
        [1.0, 3.0, 1.0],
        [4.0, 2.0, 3.0]
    ])
    svd = SVD(A)
    U, sigma, V = svd.calculate()

    # Rekonstruksi U * Sigma * V^T
    reconstructed = U @ np.diag(sigma) @ V.T
    np.testing.assert_allclose(reconstructed, A, atol=1e-5)


def test_singular_values_match_numpy():
    """Menguji bahwa nilai singular cocok dengan np.linalg.svd."""
    A = np.array([
        [4.0, 2.0, 1.0],
        [0.0, 3.0, 5.0],
        [2.0, 1.0, 4.0]
    ])
    svd = SVD(A)
    _, sigma, _ = svd.calculate()
    _, np_sigma, _ = np.linalg.svd(A)

    np.testing.assert_allclose(sigma, np_sigma, atol=1e-5)


def test_truncation():
    """Menguji fungsionalitas truncating / low-rank approximation."""
    A = np.array([
        [3.0, 1.0, 2.0],
        [1.0, 3.0, 1.0],
        [4.0, 2.0, 3.0]
    ])
    svd = SVD(A)
    
    # Rank 1 approximation
    A_k1 = svd.truncate(k=1)
    assert A_k1.shape == A.shape

    # Rank 3 approximation (Full rank) should match original
    A_k3 = svd.truncate(k=3)
    np.testing.assert_allclose(A_k3, A, atol=1e-5)

    # Invalid k values
    with pytest.raises(ValueError):
        svd.truncate(k=0)
    
    with pytest.raises(ValueError):
        svd.truncate(k=4)


def test_reconstruction_error():
    """Menguji kalkulasi error Frobenius."""
    A = np.array([
        [3.0, 1.0, 2.0],
        [1.0, 3.0, 1.0],
        [4.0, 2.0, 3.0]
    ])
    svd = SVD(A)
    
    err_k1 = svd.reconstruction_error(k=1)
    err_k2 = svd.reconstruction_error(k=2)
    err_k3 = svd.reconstruction_error(k=3)

    # Error harus semakin mengecil seiring bertambahnya rank k
    assert err_k1 >= err_k2 >= err_k3
    assert np.isclose(err_k3, 0.0, atol=1e-5)
