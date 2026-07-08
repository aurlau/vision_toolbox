import numpy as np 
from src.ops import normalize_01, normalize_uint8


def test_normalize_01_range_and_dtype(): 
  x = np.array([[0, 5], [10, 15]], dtype=np.int32) 
  y = normalize_01(x)
  assert y.dtype == np.float32
  assert float(y.min()) == 0.0 and float(y.max()) == 1.0


def test_normalize_constant_returns_zeros(): 
  x = np.full((3, 3), 7, dtype=np.int16)
  y = normalize_01(x)
  assert np.all(y == 0.0)


def test_normalize_uint8_endpoints(): 
  x = np.array([0.0, 1.0], dtype=np.float32) 
  y = normalize_uint8(x)
  assert y.dtype == np.uint8 and int(y[0]) == 0 and int(y[1]) == 255