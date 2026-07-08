"""
4個Benchmark 函數。
"""

import numpy as np

def ellipsoid(x: np.ndarray) -> float:
    """
    - Global Minimum: f(x*) = 0 at x* = (0, ..., 0)
    - Search Space: [-5.0, 5.0]^d
    """
    return float(np.dot(np.arange(1, len(x) + 1), x**2))

def rosenbrock(x: np.ndarray) -> float:
    """
    - Global Minimum: f(x*) = 0 at x* = (1, ..., 1)
    - Search Space: [-5.0, 10.0]^d
    """
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2))

def griewank(x: np.ndarray) -> float:
    """
    - Global Minimum: f(x*) = 0 at x* = (0, ..., 0)
    - Search Space: [-600.0, 600.0]^d
    """
    d = len(x)
    return float(1 + np.dot(x, x) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1)))))

def schwefel12(x: np.ndarray) -> float:
    """
    - Global Minimum: f(x*) = 0 at x* = (0, ..., 0)
    - Search Space: [-100.0, 100.0]^d
    """
    return float(sum(np.sum(x[:i+1])**2 for i in range(len(x))))

# 主程式可以動態取得函數與邊界
# Search Space可以調整
FUNC_CONFIG = {
    'ellipsoid':  {'f': ellipsoid,  'lb': -5.0,   'ub': 5.0,    'f_opt': 0.0},
    'rosenbrock': {'f': rosenbrock, 'lb': -5.0,   'ub': 10.0,   'f_opt': 0.0},
    'griewank':   {'f': griewank,   'lb': -600.0, 'ub': 600.0,  'f_opt': 0.0},
    'schwefel':   {'f': schwefel12, 'lb': -100.0, 'ub': 100.0,  'f_opt': 0.0},
}
