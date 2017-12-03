from skimage import data, io, filters,color, morphology
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
import skimage 
from skimage.feature import canny
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



### CROP 2D img y: 50:150, x: keep all
def crop_car(img):
    return img[130:155,20:320]

def crop_all(img):
    return img[50:155,20:320]


##IMAGE LABEL REGION
def img_label_all(grey_img,frameNumber):
    # apply threshold
    thresh = threshold_otsu(grey_img)
    bw = closing(grey_img > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=grey_img)

    car_count = 0 #count number of other cars coming up
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >=100 and region.area < 300:
            car_count += 1
            #draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    #show the image with detected/boxed cars
    ax.set_axis_on()
    plt.tight_layout()
    plt.show()
    plt.savefig('frame_'+str(frameNumber)+'.png')
    return car_count



##IMAGE LABEL REGION
def img_label_car(grey_img,frameNumber):
    # apply threshold
    thresh = threshold_otsu(grey_img)
    bw = closing(grey_img > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=grey_img)

    car_count = 0 #count number of other cars coming up
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area ==300:
            car_count += 1
            #draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    #show the image with detected/boxed cars
    ax.set_axis_on()
    plt.tight_layout()
    plt.show()
    plt.savefig('frame_'+str(frameNumber)+'.png')
    return car_count

##IMAGE SEGMENTATION


#filename = 'car_ahead'
#filename = 'no_cars'
#filename = 'cars'
image = io.imread(filename)
frameNumber = 1
grey_img = crop_all(color.rgb2grey(image))
edges = crop_all(filters.sobel(grey_img))
car_count = img_label_all(grey_img,frameNumber)
#print car_count
#edges = crop(canny(grey_img))
#io.imshow(edges)
#io.show()
#io.imshow(grey_img)
#io.show()
#io.imshow(grey_img - edges)
#io.show()
#plt.axis('on')
#plt.imshow(edges)
#plt.show()

#fill_cars = ndi.binary_fill_holes(edges)
#plt.imshow(fill_cars)
#io.imshow(fill_cars)
#io.show()
'''
histo = np.histogram(grey_img, bins=np.arange(0, 256))
maxVal= max([max(histo[0]),max(histo[1])])
minVal= min([min(histo[0]),min(histo[1])])
markers = np.zeros_like(grey_img)
markers[grey_img < minVal] = 1
markers[grey_img > maxVal] = 2
segmentation = morphology.watershed(filters.sobel(grey_img), markers)
#segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_objs, _ = ndi.label(segmentation)
io.imshow(segmentation)
io.show()
'''

##REMOVE BACKGROUND
'''
lightspots = np.array((edges > 245).nonzero()).T
darkspots = np.array((edges < 3).nonzero()).T
bool_mask = np.zeros(edges.shape,dtype=np.bool)
bool_mask[tuple(lightspots.T)] = True
bool_mask[tuple(darkspots.T)] = True
seed_mask, num_seeds = ndi.label(bool_mask)
ws = morphology.watershed(edges,seed_mask)
plt.imshow(ws)
io.imshow(ws)
io.show()
'''




