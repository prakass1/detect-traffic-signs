import csv
import props
import numpy as np
import cv2

# Constants
ext = ".csv"
file_const = "GT-"

def read_image(base_path=".",roi=False):
    '''
    :param base_path: The first path where all the image are placed
    :param roi: default False. Can be set to True
    :return: image_list, class_labels
    '''

    image_list = []
    class_labels = []

    print("Image Extracting started !!")

    for val in props.classes:
        # Read the file by using the prefix and base path
        reader = open(base_path + val + "//" + "".join(file_const + val + ext))
        csv_reader = csv.reader(reader, delimiter=';')
        next(csv_reader)

        for row in csv_reader:
            im = cv2.imread(base_path + val + "//" + row[0])

            if roi:
                im = im[np.int(row[4]):np.int(row[6]),
                        np.int(row[3]):np.int(row[5]),:]

            #im_resized = cv2.resize(im,(45, 45))
            image_list.append(im)
            class_labels.append(row[7])
            #print("Image Filename - ", row[0])

        #close somewhere finally
        reader.close()
        print("Extraction is now completed for class -- " + str(val) + " and current images are -- " + str(len(image_list)))

    return image_list, class_labels


def extract_features(feature, image_list):
    '''
    The function performs extraction of features on the image list.
    Implemented Features are sent in function parameter - Example: extract_features("gray", image_list) for extracting
    only grayscaled based features

    Note: This function also normalizes data, hence no need for any explicit normalization.

    :param feature:
    :param image_list:
    :return:
    '''

    #1: resize image to equal size
    resize = (32, 32)
    print("Performing resize of image to -- " + str(resize))
    X = [cv2.resize(image, resize) for image in image_list]

    #2: Normalize and mean subtraction.
    # This done to enhance local intensity of image and not look at brightness

    if feature not in ["hog", "laplacian"]:
        print("Applying normalization")
        X = np.asarray(X, dtype=np.float32)/255
        X = [x - np.mean(x) for x in X]

    if feature is not None:
        # Feature is grayscale
        if feature == "gray":
            print("Extraction for grayscale features")
            X = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in X]

        # Feature is HSV - Hue, Saturation, Value. Hue is a good feature
        if feature == "hsv":
            print("Extraction for hsv features")
            X = [cv2.cvtColor(x, cv2.COLOR_BGR2HSV) for x in X]

        # Feature is laplacian
        if feature == "laplacian":
            X = [cv2.GaussianBlur(x, (3, 3), 0) for x in X]
            X = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in X]
            # get edges
            X = [cv2.Laplacian(x, cv2.CV_64F) for x in X]

        # Feature is HoG
        if feature == "hog":
            print("Extraction for HoG features")
            # Paper params
            # (16,16) block size
            block_size = (resize[0] // 2, resize[1] // 2)
            # (8,8) cell size == block_stride (Moving cell area)
            block_stride = (resize[0] // 4, resize[1] // 4)
            cell_size = block_stride
            # number of bins - 9
            nBins = 9
            hog = cv2.HOGDescriptor(resize, block_size, block_stride, cell_size, nBins)
            X = [hog.compute(np.asarray(x, dtype=np.uint8)) for x in X]

    # Flatten and return so that it could be used in machine learning module
    print("Features are extracted, flattening the array")
    X = [x.flatten() for x in X]
    return X



def read_test_image(base_path, roi=False):
    '''
    :param base_path: The first path where all the image are placed
    :param roi: default False. Can be set to True
    :return: image_list, class_labels
    '''

    image_list = []
    #class_labels = []

    print("Image Extracting started !!")

    
        # Read the file by using the prefix and base path
    reader = open(base_path + "GT-final_test.test.csv")
    csv_reader = csv.reader(reader, delimiter=';')
    next(csv_reader)

    for row in csv_reader:
        im = cv2.imread(base_path + row[0])

        if roi:
            im = im[np.int(row[4]):np.int(row[6]),
                np.int(row[3]):np.int(row[5]),:]

        #im_resized = cv2.resize(im,(45, 45))
        image_list.append(im)
        #class_labels.append(row[7])
        #print("Image Filename - ", row[0])

        #close somewhere finally
    reader.close()
    print("Extraction is now completed for test data and number of images are -- " + str(len(image_list)))

    return image_list
