import matplotlib.image as mpimg
import numpy as np
from mlxtend.data import loadlocal_mnist




'''img = mpimg.imread("/home/etudiants/delage15u/PIDR/telecom_2019_deep_rl/classification_images/0/image_0.png")
if img.dtype == np.float32: # Si le résultat n'est pas un tableau d'entiers
    img = (img * 255).astype(np.uint8)
print("img : ",img)
'''



X, y = loadlocal_mnist(
        images_path='train-images.idx3-ubyte', 
        labels_path='train-labels.idx1-ubyte')

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])

