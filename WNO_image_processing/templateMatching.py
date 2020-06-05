from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def matchTemplate(macierz_obrazu, macierz_template):
    height_ob = macierz_obrazu.shape[0]
    width_ob = macierz_obrazu.shape[1]
    height_temp = macierz_template.shape[0]
    width_temp = macierz_template.shape[1]

    obraz_wynikowy = np.zeros((height_ob + height_temp, width_ob + width_temp))
    macierz_obrazu_zera = np.zeros((height_ob + height_temp, width_ob + width_temp))
    macierz_obrazu_zera[:-height_temp, :-width_temp] = macierz_obrazu

    for i in range(height_temp*width_temp):
        obraz_wynikowy += np.abs(np.roll(macierz_obrazu_zera, [-i // width_temp, -i % width_temp], axis=(0, 1))
                                 - macierz_template[i // width_temp, i % width_temp])

    obraz_wynikowy = obraz_wynikowy[:-height_temp, :-width_temp]
    obraz_wynikowy1 = np.asarray(obraz_wynikowy, dtype='int32')
    obraz_wynikowy1 = obraz_wynikowy1.reshape(height_ob, width_ob)
    return obraz_wynikowy1


#Wczytywanie obrazu i template
obraz = Image.open("./obrazki/liczby/liczby.png")
template = Image.open("./obrazki/liczby/template6.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

#Zamiana obrazow z RGB na skalę szarości
kolor_obraz = np.asarray(obraz, dtype='int32')
kolor_template = np.asarray(template, dtype='int32')
ax1.imshow(kolor_template, interpolation='nearest')
ax1.set_title("Template")

szary_obraz = kolor_obraz[:, :, 0]*0.2126 + kolor_obraz[:, :, 1]*0.7152 + kolor_obraz[:, :, 2]*0.0722
szary_template = kolor_template[:, :, 0]*0.2126 + kolor_template[:, :, 1]*0.7152 + kolor_template[:, :, 2]*0.0722


#Szukanie najbardziej zblizonego do template obiektu na obrazie
matched = matchTemplate(szary_obraz, szary_template)

#Szukanie poczatku bounding boxa
matched = np.ravel(matched)
indeks = np.arange(szary_obraz.shape[0]*szary_obraz.shape[1])
bounding_box = np.where(matched == np.amin(matched), indeks, 0)
bounding_box = np.trim_zeros(bounding_box)

bbox_y = np.amax(bounding_box) // szary_obraz.shape[1] + 1
bbox_x = np.amax(bounding_box) % szary_obraz.shape[1] - szary_template.shape[1]

#Wycinanie i rysowanie znalezionego template
image_cropped = kolor_obraz[bbox_y:bbox_y + szary_template.shape[0], bbox_x:bbox_x + szary_template.shape[1], :]
ax2.imshow(image_cropped, interpolation='nearest')
ax2.set_title("Znaleziony")

#Rysowanie obrazu z bounding boxem
fig2, ax21 = plt.subplots(1, 1, figsize=(12, 8))

ax21.imshow(kolor_obraz, interpolation='nearest')
ax21.plot((bbox_x, bbox_x), (bbox_y, bbox_y + szary_template.shape[0]), color='green')
ax21.plot((bbox_x, bbox_x + szary_template.shape[1]), (bbox_y, bbox_y), color='green')
ax21.plot((bbox_x + szary_template.shape[1], bbox_x + szary_template.shape[1]), (bbox_y, bbox_y + szary_template.shape[0]), color='green')
ax21.plot((bbox_x, bbox_x + szary_template.shape[1]), (bbox_y + szary_template.shape[0], bbox_y + szary_template.shape[0]), color='green')


plt.show()
