############### template matching ####################

import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

def match_template(img, templates_location, threshold):
    img = cv2.imread(img, cv2.COLOR_BGR2RGB)

    w, h = img[:,:,0].shape[::-1]

    if w > 400 and h > 400:
        print("Resizing image to specific size")
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
    template_sizes = np.arange(40, 160, 10)
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

            w, h = thresh_temp[:,:,1].shape[::-1]


            # Apply template Matching
            thresh = threshold
            res = cv2.matchTemplate(thresh_img[:,:,1], thresh_temp[:,:,1], eval(method))

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
            #Write the image
            cv2.imwrite("images/to_pred_img.png", cropped)
            cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 4)

            plt.subplot(121), plt.imshow(cropped, cmap='gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(img_copy, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(method)
            plt.show()
            if cropped is None:
                return False
            else:
                return cropped
            #plt.savefig("match-" + str(count) + ".png")


#### A Wrapper function
def do_tm(img_loc, temp_loc):
    min_thresh = 0.45
    max_thresh = 0.95
    match_img_list = []
    while max_thresh >= min_thresh:
        match_img = match_template(img_loc,
                                   temp_loc, max_thresh)
        if match_img is not False and match_img is not None:
            match_img_list.append(match_img)
            break
        max_thresh = max_thresh - 0.10

    if len(match_img_list) > 0:
        return match_img_list[0]

######## Main function to test
# min_thresh = 0.45
# max_thresh = 0.95
# match_img_list = []
# while max_thresh >= min_thresh:
#     match_img = match_template("templates//images//zigzag.jpg",
#                                 "templates//*.png",max_thresh)
#     if match_img is not False and match_img is not None:
#         match_img_list.append(match_img)
#         break
#     max_thresh = max_thresh - 0.10
#
# if len(match_img_list) > 0:
#     import machine_learning as ml
#     feature = "hog"
#     prediction = ml.make_single_img_prediction(feature, match_img_list[0])
#     pred_class = ml.class_switcher(str(prediction[0]))
#
#     fig = plt.figure(dpi=100, tight_layout=True, frameon=False,
#                      figsize=(10,8))  # dpi & figsize of my choosing
#     fig.figimage(match_img_list[0], cmap=plt.cm.binary)
#     plt.imsave(arr = match_img_list[0], fname="images//" + "tm_" + feature)
#     plt.text(0.5, 0.5, str(prediction[0]) + " - " + pred_class, horizontalalignment='left', verticalalignment='top', color="g",
#              weight="bold")
#     location_pred = "images//" + "tm_" + feature + "prediction.png"
#     plt.savefig("images//" + "tm_" + feature + "prediction.png")
#     plt.close()
#     print("Predictions saved at - ", str(location_pred))

