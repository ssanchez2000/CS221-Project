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
import math


'''
def crop_car(img):
    return img[130:155,20:320]

def crop_all(img):
    return img[50:155,20:320]

def crop_car(img):
    return img[80:155,20:320]

##IMAGE LABEL REGION
def img_label_all(grey_img,frameNumber,rgb_img):
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
        if region.area >=50 and region.area<300:
            #draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
	    #print minr,minc,maxr,maxc
            pixel=[int((maxr+minr)/2),int((maxc+minc)/2)]
	    car_pixel = list(rgb_img[pixel[0],pixel[1],:])
	    if set(car_pixel) != set(agent_pixel):
	        car_count += 1
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    #show the image with detected/boxed cars
    ax.set_axis_on()
    plt.tight_layout()
    plt.show()
    plt.savefig('frame_'+str(frameNumber)+'.png')
    return car_count+1



##IMAGE LABEL REGION
def img_label_car(grey_img,frameNumber,rgb_img):
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
        if region.area >=300:
            car_count += 1
            #draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
	    #print minr,minc,maxr,maxc
	    pixel=[int((maxr+minr)/2),int((maxc+minc)/2)]
	    #get agent's color pixel
	    #print rgb_img[pixel[0],pixel[1],:]
	    io.imshow(rgb_img)
            io.show()
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    #show the image with detected/boxed cars
    ax.set_axis_on()
    plt.tight_layout()
    plt.show()
    plt.savefig('frame_'+str(frameNumber)+'.png')
    return car_count


'''

agent_pixel = [192, 192, 192]

def crop(img):
    return img[75:156,20:320]

##IMAGE LABEL REGION
def img_label(grey_img,frameNumber,rgb_img):
    # apply threshold
    thresh = threshold_otsu(grey_img)
    bw = closing(grey_img > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=grey_img)

    car_count = 0 #count number of other cars coming up
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >=50 and region.area<300:
            #draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
	    #print minr,minc,maxr,maxc
            pixel=[int((maxr+minr)/2),int((maxc+minc)/2)]
	    car_pixel = list(rgb_img[pixel[0],pixel[1],:])
	    if set(car_pixel) != set(agent_pixel):
	        car_count += 1
    #plus an additional 1 to place our agent
    return car_count+1


##IMAGE SEGMENTATION


####
# our agents color pixel is [192 192 192]
###

#filename = 'car_ahead'
#filename = 'no_cars'
#filename = 'cars'
#filename ='frame_1.png'
#filename ='frame_81.png'
#filename ='frame_179.png'
image = io.imread(filename)
frameNumber = 0.0
grey_img = crop(color.rgb2grey(image))
#grey_img = crop_car(color.rgb2grey(image))
#edges = crop_all(filters.sobel(grey_img))
car_count = img_label(grey_img,frameNumber,crop(image))
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




