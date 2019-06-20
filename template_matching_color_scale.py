############### template matching ####################

import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

def match_template(img, templates_location, threshold):
    img = cv2.imread(img, cv2.COLOR_BGR2RGB)

    #img = cv2.imread("D://Jannes//Dokumente//Uni//introCV/templates/test-image.jpg", cv2.COLOR_BGR2RGB)

    w, h = img[:,:,0].shape[::-1]

    if w > 400 and h > 400:
        print("Came here")
        img = cv2.resize(img, (400, 400))

    blurred_frame = cv2.GaussianBlur(img, (5, 5), 0)
    img_hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2HSV)
    #cv2.imshow("hsv", img_hsv)

    # Every color except white
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv2.inRange(img_hsv, low, high)
    result = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
   # cv2.imshow("res", result)

    # HSV -
    # Green color
    low_red = np.array([25, 52, 72])
    #low_green = np.array([15,55,25])
    high_red = np.array([180, 255, 255])

    green_mask = cv2.inRange(img_hsv, low_red, high_red)
    #cv2.imshow("mask", green_mask)
    green_img = cv2.bitwise_and(img_hsv, img_hsv, mask=green_mask)
    #cv2.imshow("mask",green_img)

    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    #mask = cv2.inRange(img_hsv, (45, 25, 25), (65, 255,255))
    #green_img = cv2.bitwise_and(img_hsv, img_hsv, mask=green_mask)
    #cv2.imshow("mask", green_img)

    imgray = cv2.cvtColor(green_img, 0)
    ret,thresh_img = cv2.threshold(imgray, 115, 255, 0)
    #cv2.imshow("thresh", thresh_img[:,:,2])


    # Template match
    # Read the template

    method = "cv2.TM_CCOEFF_NORMED"
    #template = cv2.imread("C://Users//subash//Desktop//templates//diamond-template.png", cv2.COLOR_BGR2RGB)

    #different template sizes, change as needed
    template_sizes = np.arange(40, 75, 5)
    template_names = glob.glob(templates_location)
    templates = np.array([np.array(cv2.imread(name, cv2.COLOR_BGR2RGB)) for name in template_names])
    for template in templates:
        for template_size in template_sizes:
            template = cv2.resize(template, (template_size, template_size))
            template_hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
            blurred_frame = cv2.GaussianBlur(template_hsv, (5, 5), 0)
            #cv2.imshow("bluu", blurred_frame)
            low = np.array([0, 42, 0])
            high = np.array([179, 255, 255])
            #low_green = np.array([2,55,50])
            #high_green = np.array([180, 255, 255])

            low_red = np.array([25, 52, 72])
            high_red = np.array([150, 255, 255])
            green_mask = cv2.inRange(blurred_frame, low_red, high_red)
            green_template = cv2.bitwise_and(blurred_frame, blurred_frame, mask=green_mask)
           # cv2.imshow("template", green_template)
            temp_gray = cv2.cvtColor(green_template, 0)
            ret, thresh_temp = cv2.threshold(temp_gray, 135, 200, 0)
            #cv2.imshow("thresh_temp", thresh_temp[:,:,2])

            #ret, thresh_t = cv2.threshold(template, 50, 100, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            #kernel = np.ones((3,3) , np.int8)
            #dila = cv2.(thresh_t, kernel)
            #plt.imshow(dila, cmap="gray")
            # ret, thresh_temp = cv2.threshold(template, 50, 100, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # plt.imshow(thresh1, cmap="gray")
            #t_edges = cv2.Canny(template, 100, 150)
            # ret,thresh1 = cv2.threshold(img_hue, 50, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # plt.imshow(thresh1, cmap="gray")
            #kernel = np.ones((5, 5), np.uint8)
            #t_edges = t_edges - cv2.dilate(t_edges, kernel)
            # plt.imshow(opening, cmap="gray")
            # blurred = cv2.GaussianBlur(opening,(3,3),0)
            # plt.imshow(edges, cmap="gray")

            # template = cv2.resize(template,(55, 60))
            # template = cv2.GaussianBlur(opening, (3, 3), 0)
            # plt.savefig(str(i) + "_img.png")
            # plt.imshow(template, cmap="gray")
            # template_edges = cv2.Canny(template,100,200)
            # plt.imshow(template_edges, cmap="gray")
            w, h = thresh_temp[:,:,1].shape[::-1]

            # All the 6 methods for comparison in a list
            # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

            # Apply template Matching
            thresh = threshold
            res = cv2.matchTemplate(thresh_img[:,:,1], thresh_temp[:,:,1], eval(method))
            #loc = np.where(res >= thresh)
            #if loc[0].size != 0:
            #print(loc)
            #print("There is a match")
            #match_count = match_count + 1
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            #check value of match, adjust if minimum is used
            if max_val < thresh:
                continue
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)

            print(top_left)
            print(bottom_right)
            #crop array in region top_left->bottom_right
            img_copy = img.copy()
            cropped = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 4)

            #plt.subplot(121), plt.imshow(cropped, cmap='gray')
            #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            #plt.subplot(122), plt.imshow(img, cmap='gray')
            #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            #plt.suptitle(method)
            #plt.show()
            if cropped is None:
                return False
            else:
                return cropped
            #plt.savefig("match-" + str(count) + ".png")


######## Test function

min_thresh = 0.45
max_thresh = 0.95
match_img_list = []
while max_thresh >= min_thresh:
    match_img = match_template("C://Users//subash//Desktop//templates//images//1.jpg",
                                "templates//*.png",max_thresh)
    if match_img is not False and match_img is not None:
        match_img_list.append(match_img)
        break
    max_thresh = max_thresh - 0.10

if len(match_img_list) > 0:
    import machine_learning as ml
    ml.make_single_img_prediction("hog", match_img_list[0])

