from PIL import Image
from pylab import *
from math import sin

def histogram_szary(sing, ax3, ax4):
    iar = np.asarray(sing, 'int32')

    gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(int32)
    gray = gray(iar)

    D = [np.count_nonzero(gray == x) for x in range(0, 255)]

    ax3.plot(D)
    ax4.imshow(gray, cmap=plt.get_cmap(name='gray'))

    return Image.fromarray(gray)

sing = Image.open('lena.png').convert('RGB')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
img = histogram_szary(sing, ax3, ax4)

iar = np.asarray(img, 'int32')

image_padded = np.zeros((iar.shape[0] + 1, iar.shape[1] + 1), 'int32')
image_padded[0:-1, 0:-1] = iar

image_up = np.zeros((iar.shape[0] + 1, iar.shape[1] + 1), 'int32')
image_up[1:, 1:] = iar

Gx = image_up - image_padded
Gy = image_padded - image_up
G = np.sqrt(Gx ** 2 + Gy ** 2)
print(G.shape)

ax1.imshow(G, cmap=plt.get_cmap(name='gray'))

ax2.imshow(Gx, cmap=plt.get_cmap(name='gray'))
show()