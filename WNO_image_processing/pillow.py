from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

roberts_cross_v = np.array( [[ 1, 0 ],
                             [ 0,-1 ]] )

roberts_cross_h = np.array( [[0, 1 ],
                             [-1, 0 ]] )

def getRed(redVal):
    return '#%02x%02x%02x' % (redVal, 0, 0)
def getGreen(greenVal):
    return '#%02x%02x%02x' % (0, greenVal, 0)
def getBlue(blueVal):
    return '#%02x%02x%02x' % (0, 0, blueVal)

def roberts_cross( imag ):
    image5 = np.asarray( imag, dtype='int32' )
    vertical = ndimage.convolve( image5, roberts_cross_v )
    horizontal = ndimage.convolve( image5, roberts_cross_h )
    output_image = np.sqrt( np.square(horizontal) + np.square(vertical))
    #output_image = np.abs(horizontal) + np.abs(vertical)
    img = Image.fromarray(np.asarray(np.clip(output_image, 0, 255), dtype="uint8"), "L")
    img.show()
    #img.save("img1.png")

# Create an Image with specific RGB value
image = Image.open("lena.png")
# Display the image
image.show()
# Get the color histogram of the image
histogram = image.histogram()
# Take only the Red counts
l1 = histogram[0:256]
# Take only the Blue counts
l2 = histogram[256:512]
# Take only the Green counts
l3 = histogram[512:768]
plt.figure(0)
# R histogram
for i in range(0, 256):
    plt.bar(i, l1[i], color=getRed(i), edgecolor=getRed(i), alpha=0.5)
    plt.bar(i, l2[i], color=getGreen(i), edgecolor=getGreen(i), alpha=0.5)
    plt.bar(i, l3[i], color=getBlue(i), edgecolor=getBlue(i), alpha=0.5)

image1 = image.convert("L")
histogram = image1.histogram()
plt.figure(1)
for i in range(0, 256):
    plt.bar(i, histogram[i], color='gray', edgecolor='gray', alpha=0.5)
image1.show()

im = image1.load()
roberts_cross(image1)


plt.show()