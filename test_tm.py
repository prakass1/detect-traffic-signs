##### Some testing functions here which will be removed later
#Main
#image_list, class_labels = read_image(base_path="F://Modules_Courses//Semester-4//Computer Vision//data//GTSRB//Final_Training//Images//")
#print("Length of image ", len(image_list))

#X = extract_features("hsv", image_list)


# 5;6;24;25

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

import matplotlib.pyplot as plt
import cv2
import numpy as np
img_hue = cv2.imread("C://Users//subash//Desktop//templates//image.jpg",0)
edges = cv2.Canny(img_hue,250,500)
#ret,thresh1 = cv2.threshold(img_hue, 50, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#plt.imshow(thresh1, cmap="gray")
kernel = np.ones((5,5), np.uint8)
edges = edges - cv2.dilate(edges,kernel)
#plt.imshow(opening, cmap="gray")
#blurred = cv2.GaussianBlur(opening,(3,3),0)
#plt.imshow(edges,cmap="gray")
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold

#plt.imshow(edges, cmap="gray")
#ret, thresh = cv2.threshold(edges ,125,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#plt.imshow(cv2.drawContours(img_hue, contours, 0, (0, 255, 0), 3), cmap="gray")


#Template match
# Read the template
import processing as pp
import props as properties
import numpy as np
imgs, class_labels = pp.read_image(properties.train_base_dir, roi=True)

unique_classes, counts = np.unique(np.asarray(class_labels, dtype=np.int8), return_counts=True)
freq_count = np.asarray((unique_classes, counts)).T

img_dict ={}
image_count = 0
for freq in freq_count:

    if img_dict:
        image_count = image_count + len(img_dict[prev_idx])
        img_dict[freq[0]] = imgs[image_count + 1 : image_count + freq[1]]
        prev_idx = freq[0]
    else:
        img_dict[freq[0]] = imgs[0:freq[1]]
        prev_idx = freq[0]

count=0
match_count = 0
method = "cv2.TM_CCOEFF_NORMED"
for key, vals in img_dict.items():
    print("Searching for templates from class - " + str(key))

    for i in range(len(vals) - 50, len(vals)):

        template = cv2.cvtColor(vals[i], cv2.COLOR_BGR2GRAY)
        #ret, thresh_temp = cv2.threshold(template, 50, 100, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #plt.imshow(thresh1, cmap="gray")
        t_edges = cv2.Canny(template, 100, 150)
        # ret,thresh1 = cv2.threshold(img_hue, 50, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # plt.imshow(thresh1, cmap="gray")
        kernel = np.ones((5, 5), np.uint8)
        t_edges = t_edges - cv2.dilate(t_edges, kernel)
        # plt.imshow(opening, cmap="gray")
        # blurred = cv2.GaussianBlur(opening,(3,3),0)
        #plt.imshow(edges, cmap="gray")

        #template = cv2.resize(template,(55, 60))
        #template = cv2.GaussianBlur(opening, (3, 3), 0)
        #plt.savefig(str(i) + "_img.png")
        #plt.imshow(template, cmap="gray")
    #template_edges = cv2.Canny(template,100,200)
    #plt.imshow(template_edges, cmap="gray")
        w, h = t_edges.shape[::-1]


# All the 6 methods for comparison in a list
#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# Apply template Matching
        thresh = .40
        res = cv2.matchTemplate(edges, t_edges, eval(method))
        loc = np.where(res >= thresh)
        if loc[0].size != 0:
            print(loc)
            print("There is a match")
            match_count = match_count + 1
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            print(top_left)
            print(bottom_right)

            cv2.rectangle(img_hue, top_left, bottom_right, (0,255,0) ,4)

            plt.subplot(121), plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img_hue,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle("cv2.TM_CCOEFF")
            plt.show()
            plt.savefig("match-" + str(count) + ".png")
            count+=count

        #if (match_count > 3):
         #   print("Breaking")
          #  break

    #if(match_count > 3):
     #   print("Breaking")
      #  break

'''
ima = cv2.imread("C://Users//subash//Desktop//image.jpg",0)

#Template match
# Read the template
template = cv2.imread('C://Users//subash//Desktop//template1.png', 0)
template = cv2.resize(template,(150,150))
cv2.imshow("im",template)
w, h = template.shape[::-1]


# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

img = ima.copy()



# Apply template Matching
res = cv2.matchTemplate(img,template,eval("cv2.TM_CCOEFF_NORMED"))
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

print(top_left)
print(bottom_right)

cv2.rectangle(img, top_left, bottom_right, 255, 4)

plt.subplot(121), plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle("cv2.TM_CCOEFF")
plt.show()


temp = ima.copy()
#Y to X and not X to Y
temp = temp[133:283,780:930]
plt.imshow(temp)
plt.savefig("C://Users//subash//Desktop//template1.png")

#ret, thresh = cv2.threshold(gray[:,:,2],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#plt.imshow(thresh)

#im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#plt.imshow(cv2.drawContours(image_list[6889], [contours[4]], 0, (0,0,255),3))


#Visualize an image
#Thresholding
#shape classification (This has some problem with respect to our dataset)
#ROI is given in test. Ideally, for basic scenario it should be okay to apply straight


#Things:
# 1. Try the paper formula on an image
# 2. See if shape is generated as expected
'''

'''
edges = cv2.Canny(image_list[6889],50,100)

plt.subplot(121),plt.imshow(image_list[6889],cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

## Contour
im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i in contours:
    plt.imshow(cv2.drawContours(image_list[6889], contours, i, (0,0,255),hierarchy, 2, 8, 0))

'''

### Some visualization - Ideal to move this to a separate file if required
'''
fig, ax = plt.subplots(6,6,figsize=(15,8))
plt.tight_layout()
for i in range(0,6):
    for j in range(0,6):
        randInt = np.random.randint(0, len(image_list))
        ax[i][j].imshow(image_list[randInt], cmap="gist_gray")
plt.savefig("plt_traffic_sign.png")
'''