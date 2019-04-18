
import load
import enhance

from PIL import Image
import numpy as np
from matplotlib import pylab as plt

verbose = True

if __name__ == "__main__":
    
    images = load.fingerprints("DB/Rindex28")

    if (verbose):
        print(type(images[0]))
        print('max', np.array(images[0]).max())
        print('min', np.array(images[0]).min())

    images_enhanced = []

    for image in images[:1]:
        image = enhance.contrast(image)
        image = enhance.median_filter(image, 5)
        images_enhanced.append(image)

    image = images_enhanced[0]

    if (verbose):
        print("max", image.max())
        print("min", image.min())
        plt.imshow(Image.fromarray(image), cmap='gray')
        plt.show()
