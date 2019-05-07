'''
This class will ideally do initial bit to extract the csvs of the required classes
and extract key features to create a data to feed into machine learning models

Authors:


Running system:
python 3.6

'''

import csv
import props
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
ext = ".csv"
file_const = "GT-"
def read_image(base_path=".",roi=False):
        image_list = []
        class_labels = []

        for val in props.classes:
            # Read the file by using the prefix and base path
            reader = open(base_path + val + "//" + "".join(file_const + val + ext))
            csv_reader = csv.reader(reader, delimiter=';')
            next(csv_reader)

            for row in csv_reader:
                im = cv2.imread(base_path + val + "//" + row[0])
                image_list.append(im)
                class_labels.append(row[7])
                print("Image Filename - ", row[0])

            #close somewhere finally
            reader.close()
        return image_list, class_labels

#Main
image_list, class_labels = read_image(base_path="")
print("Length of image ", len(image_list))


