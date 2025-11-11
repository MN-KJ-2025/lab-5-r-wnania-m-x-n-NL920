# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np


def spare_matrix_Abt(m: int, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja tworząca zestaw składający się z macierzy A (m,n) i
    wektora b (m,) na podstawie pomocniczego wektora t (m,).

    Args:
        m (int): Liczba wierszy macierzy A.
        n (int): Liczba kolumn macierzy A.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            - Macierz A o rozmiarze (m,n),
            - Wektor b (m,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(m, int) or not isinstance(n, int):
        return None
    if m <= 0 or n <= 0:
        return None
    t=np.linspace(0,1,m)
    b=np.cos(4*t)
    A=np.vander(t, N=n, increasing=True)
    return (A,b)


def square_from_rectan(
    A: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja przekształcająca układ równań z prostokątną macierzą współczynników
    na kwadratowy układ równań.
    A^T * A * x = A^T * b  ->  A_new * x = b_new

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej stronie równania.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            - Macierz A_new o rozmiarze (n,n),
            - Wektor b_new (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        # --- Walidacja typów ---
        if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
            return None
        if A.ndim != 2 or b.ndim not in [1, 2]:
            return None

        # --- Dopasowanie wymiarów ---
        m, n = A.shape
        b = b.reshape(-1)  # konwertuj na wektor 1D
        if b.shape[0] != m:
            return None

        # --- Obliczenia ---
        A_new = A.T @ A
        b_new = A.T @ b

        # --- Walidacja wyniku ---
        if not isinstance(A_new, np.ndarray) or not isinstance(b_new, np.ndarray):
            return None

        return (A_new, b_new)  # MUSI być krotka, nie lista ani macierz

    except Exception:
        return None
    


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej stronie równania.

    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
            return None
        if A.ndim != 2 or x.ndim != 1 or b.ndim != 1:
            return None
        m, n = A.shape
        if len(x) != n or len(b) != m:
            return None
        return np.linalg.norm(A @ x - b)
    except Exception:
        return None
    return np.linalg.norm(A @ x - b)

