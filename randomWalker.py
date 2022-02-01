from skimage import  io, img_as_float
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import  exposure
from skimage.segmentation import random_walker
img = img_as_float(io.imread('1.png'))

sigmaEst = np.mean(estimate_sigma(img, multichannel=True))
pathKW = dict(patch_size=5,
              patch_distance=6,
              multichannel=True)

denoiseImg = denoise_nl_means(img, h=1.15*sigmaEst, fast_mode=True, **pathKW)
histEqualizeImg = exposure.equalize_adapthist(denoiseImg)
plt.imshow(histEqualizeImg, cmap='gray')
#io.imshow(histEqualizeImg)
#plt.hist(histEqualizeImg.flat, bins=100, range=(0,1))
plt.show()

markers = np.zeros(img.shape, dtype=np.uint)
markers[(histEqualizeImg >= 0) & (histEqualizeImg < 0.4)] = 1
markers[(histEqualizeImg >= 0.4) & (histEqualizeImg < 0.9)] = 2

#plt.imshow(markers)
#plt.show()
labels = random_walker(histEqualizeImg, markers, beta=30, mode='cg_j')

segm1 = (labels==1)
segm2 = (labels==2)

allSegments = np.zeros((histEqualizeImg.shape[0], histEqualizeImg.shape[1], 3))

allSegments[segm1] = (0,1,0)
allSegments[segm2] = (1,0,0)
plt.imsave('segmented_image.png', allSegments)
plt.imshow(allSegments)
plt.show()