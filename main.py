import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut,colorwheel,coffee,coins,camera
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import skimage.color as color
#%%
img = img_as_float(astronaut()[::2, ::2])

segments_fz = felzenszwalb(img, scale=30, sigma=0.5, min_size=10)
segments_slic = slic(img, n_segments=img.shape[0]**2/64, compactness=10, sigma=1)
segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=250, compactness=0.001)

print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()

#%%
plt.imshow(color.label2rgb(segments_fz, img, kind='avg'))
#%%

n = len(np.unique(segments_fz))
A = np.zeros((n,n))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pixel = segments_fz[i,j]
        if i > 0:
            adj = segments_fz[i-1,j]
            if adj != pixel:
                A[pixel, adj] = 1
                A[adj, pixel] = 1
        if i < img.shape[0]-1:
            adj = segments_fz[i+1,j]
            if adj != pixel:
                A[pixel, adj] = 1
                A[adj, pixel] = 1
        if j > 0:
            adj = segments_fz[i,j-1]
            if adj != pixel:
                A[pixel, adj] = 1
                A[adj, pixel] = 1
        if j < img.shape[1]-1:
            adj = segments_fz[i,j+1]
            if adj != pixel:
                A[pixel, adj] = 1
                A[adj, pixel] = 1

#%%

fig, ax = plt.subplots()
ax.imshow(color.label2rgb(segments_fz[:30,:30], img[:30,:30], kind='avg'))
for i in range(30):
    for j in range(30):
        text = ax.text(j, i, segments_fz[i, j],
                       ha="center", va="center", color="r")


