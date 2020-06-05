from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def splot(macierz_obrazu, macierz_maski):
    height1 = macierz_obrazu.shape[0]
    width1 = macierz_obrazu.shape[1]
    n = macierz_maski.shape[0]
    obraz_wynikowy = np.zeros((height1 + n, width1 + n))
    macierz_obrazu_zera = np.zeros((height1 + n, width1 + n))
    macierz_obrazu_zera[:-n, :-n] = macierz_obrazu

    for i in range(n*n):
        obraz_wynikowy += np.roll(macierz_obrazu_zera, [-i//n, -i%n], axis=(0,1)) * macierz_maski[i//n, i%n]

    obraz_wynikowy = obraz_wynikowy[n:-2*n, n:-2*n]
    obraz_wynikowy1 = np.asarray(obraz_wynikowy, dtype='int32')
    obraz_wynikowy1 = obraz_wynikowy1.reshape(height1 - 2*n, width1 - 2*n)
    return obraz_wynikowy1


#Wczytywanie i rysowanie obrazu
obraz = Image.open("./obrazki/kar/kar1.jpg")
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))
ax1.imshow(obraz)
ax1.set_title("Obraz niezmodyfikowany")

#Zamiana obrazu z RGB na skalę szarości
macierz_obraz = np.asarray(obraz, dtype="int32")

czerwony = macierz_obraz[:, :, 0]
zielony = macierz_obraz[:, :, 1]
niebieski = macierz_obraz[:, :, 2]

szary_macierz = czerwony*0.2126 + zielony*0.7152 + niebieski*0.0722
szary_macierz = szary_macierz.astype(int)
szary4_macierz = szary_macierz // 4

#Rysowanie szarego obrazu
ax2.imshow(Image.fromarray(szary_macierz))
ax2.set_title("Szary obraz")

#Wyliczanie histogramu szarości
element, ile_razy = np.unique(szary4_macierz, return_counts=True)
histogram_szary = np.zeros(64, dtype='int32')
histogram_szary[element] = ile_razy
wartosci_histogram = np.arange(64)

#Rysowanie histogramu szarości
ax3.bar(wartosci_histogram, histogram_szary, color='gray')
ax3.set_title("Histogram skali szarosci")

#Tworzenie filtru rozmycia Gaussowskiego
roz = 9  #wymiar filtru
sigma = 0.7
filtr1 = np.arange(roz*roz).reshape(roz, roz)
gaussian_blur = np.zeros((roz, roz))
gaussian_blur = np.exp(-1*(np.square(filtr1%roz - roz//2) + np.square(filtr1//roz - roz//2))/(2*np.square(sigma)) ) \
                / (2*np.pi*np.square(sigma))

#Splatanie obrazu z filtrem rozmycia i jego rysowanie
obraz_gaussian = splot(szary_macierz, gaussian_blur)
ax4.imshow(Image.fromarray(obraz_gaussian))
ax4.set_title("Obraz rozmycie gaussowskie")

#Macierze krzyza Robertsa
roberts_cross_v = np.array([[1, 0],
                            [0,-1]])
roberts_cross_h = np.array([[0, 1],
                            [-1,0]])

#Obliczanie splotu krzyza Robertsa z obrazem
# obraz_roberts_v = obraz_gaussian - np.roll(obraz_gaussian, [1, 1], axis=(0, 1))
# obraz_roberts_h = -np.roll(obraz_gaussian, 1, axis=0) + np.roll(obraz_gaussian, 1, axis=1)
obraz_roberts_v = splot(obraz_gaussian, roberts_cross_v)
obraz_roberts_h = splot(obraz_gaussian, roberts_cross_h)
obraz_roberts = np.sqrt(np.square(obraz_roberts_h) + np.square(obraz_roberts_v))

#Prewitt
# prewitt_x = np.array( [[1, 0, -1 ],
#                        [1, 0, -1 ],
#                        [1, 0, -1 ]] )
# prewitt_y = np.array( [[1, 1, 1 ],
#                        [0, 0, 0 ],
#                        [-1, -1, -1 ]] )


#Wczytanie i rysowaniie obrazu przepuszczonego przez filtr wykrywajacy krawedzie
obraz_krawedzie = obraz_roberts
obraz_krawedzie_x = obraz_roberts_v
obraz_krawedzie_y = obraz_roberts_h
ax5.imshow(Image.fromarray(obraz_krawedzie))
ax5.set_title("Obraz wykrywanie krawedzi")

#Wyliczanie gradientow kierunkowych zmian
grad = np.degrees(np.arctan2(obraz_krawedzie_x, obraz_krawedzie_y))

#Zaokraglanie gradientow kierunkowych do wartosci 0, 45, 90, 135
grad_rounded = np.zeros(grad.shape[0]*grad.shape[1]).reshape(grad.shape[0], grad.shape[1])
grad_rounded = np.where(np.logical_or(np.logical_and(grad >= 22.5, grad < 67.5), np.logical_and(grad >= -157.5, grad < -112.5)),
                        45, grad_rounded)
grad_rounded = np.where(np.logical_or(np.logical_and(grad >= 67.5, grad < 112.5), np.logical_and(grad >= -112.5, grad < -67.5)),
                        90, grad_rounded)
grad_rounded = np.where(np.logical_or(np.logical_and(grad >= 112.5, grad < 157.5), np.logical_and(grad >= -67.5, grad < -22.5 )),
                        135, grad_rounded)

#Wyznaczanie potencjalnych krawedzi na podstawie zaokraglonych gradientow kierunkowych
grad1 = np.zeros((grad.shape[0], grad.shape[1]))
grad1 = np.where(np.logical_and(grad_rounded == 0, obraz_krawedzie > np.roll(obraz_krawedzie, 1, axis=1),
                                obraz_krawedzie > np.roll(obraz_krawedzie, -1, axis=1)), 1, grad1)
grad1 = np.where(np.logical_and(grad_rounded == 90, obraz_krawedzie > np.roll(obraz_krawedzie, 1, axis=0),
                                obraz_krawedzie > np.roll(obraz_krawedzie, -1, axis=0)), 1, grad1)
grad1 = np.where(np.logical_and(grad_rounded == 45, obraz_krawedzie > np.roll(obraz_krawedzie, [1, 1], axis=(0, 1)),
                                obraz_krawedzie > np.roll(obraz_krawedzie, [-1, -1], axis=(0, 1))), 1, grad1)
grad1 = np.where(np.logical_and(grad_rounded == 135, obraz_krawedzie > np.roll(obraz_krawedzie, [1, -1], axis=(0, 1)),
                                obraz_krawedzie > np.roll(obraz_krawedzie, [-1, 1], axis=(0, 1))), 1, grad1)

#Progowanie krawedzi petencjalnych
th = 0.07
tl = 0.03
obraz_progowanie = np.zeros((grad.shape[0], grad.shape[1]))
obraz_progowanie = np.where(np.logical_and(obraz_krawedzie >= th*255, grad1 == 1), 2, obraz_progowanie)
obraz_progowanie = np.where(np.logical_and(obraz_krawedzie >= tl*255, obraz_krawedzie < th*255, grad1 == 1),
                            1, obraz_progowanie)

#Zmiana wartosci krawedzi o progu wysokim i sprawdzanie o progu niskim
obraz_canny = np.zeros((grad.shape[0], grad.shape[1]))
obraz_canny = np.where(obraz_progowanie == 2, 255, obraz_canny)
obraz_canny = np.where(np.logical_and(obraz_progowanie == 1,
                                      np.logical_or (np.roll(obraz_progowanie, [-1, 1], axis=(0, 1)) == 2,
                                        np.roll(obraz_progowanie, [0, 1], axis=(0, 1)) == 2,
                                        np.roll(obraz_progowanie, [1, 1], axis=(0, 1)) == 2)), 255, obraz_canny)
obraz_canny = np.where(np.logical_and(obraz_progowanie == 1,
                                      np.logical_or (np.roll(obraz_progowanie, [-1, 0], axis=(0, 1)) == 2,
                                        np.roll(obraz_progowanie, [1, 0], axis=(0, 1)) == 2)), 255, obraz_canny)
obraz_canny = np.where(np.logical_and(obraz_progowanie == 1,
                                      np.logical_or (np.roll(obraz_krawedzie, [-1, -1], axis=(0, 1)) == 2,
                                        np.roll(obraz_progowanie, [0, -1], axis=(0, 1)) == 2,
                                        np.roll(obraz_progowanie, [1, -1], axis=(0, 1)) == 2)), 255, obraz_canny)


#Rysowanie koncowego obrazu
ax6.imshow(Image.fromarray(obraz_canny))
ax6.set_title("Obraz filtr Canny'ego")


plt.show()
