"""SVD (Singular Value Decomposition) from Scratch.

Package untuk menghitung dekomposisi nilai singular dari matriks,
rekonstruksi low-rank (truncation), dan estimasi error Frobenius.
"""

from .core import SVD

__all__ = ["SVD"]
