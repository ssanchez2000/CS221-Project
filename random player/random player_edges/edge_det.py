from skimage import data, io, filters,color,morphology
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
import skimage 
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches




#filename = os.path.join(skimage.data_dir,'cars.png')
filename = 'cars'
image = io.imread(filename)
grey_img = color.rgb2grey(image)
edges = filters.sobel(grey_img)
#io.imshow(edges)
#io.show()
#io.imshow(grey_img)
#io.show()
#io.imshow(grey_img - edges)
#io.show()



#fill_cars = ndi.binary_fill_holes(edges)
#plt.imshow(fill_cars)


#bool_mask = np.zeros(edges.shape,dtype=np.bool)

#bool_mask[tuple(lightspots.T)] = True
#bool_mask[tuple(darkspots.T)] = True
#seed_mask, num_seeds = ndi.label(bool_mask)
#ws = morpholoy.watershed(edges,seed_mask)
#plt.imshow(ws)


# apply threshold
thresh = threshold_otsu(grey_img)
bw = closing(grey_img > thresh, square(3))

# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
image_label_overlay = label2rgb(label_image, image=grey_img)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()

