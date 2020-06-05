from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

macierz = 100*np.random.rand(10000).reshape((100, 100))
bounding_box1 = np.where(macierz == np.ndarray.min(macierz), 1, 0)

print("koniec")

