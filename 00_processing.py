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
import pandas as pd

ext = ".csv"
file_const = "GT-"
def read_image(base_path=".",roi=False):
        image_list = []
        class_labels = []

        for val in props.classes:
            with open(base_path+val+"\\"+val+ext) as f:
                csv_reader = csv.reader(f, delimiter=';')
                next(f)
                for row in csv_reader:
                    row[0] = base_path + val + row[0]
                    image_list.append(row)
            
        dataFrame = pd.DataFrame(image_list, columns=['Filename','Width','Height','Roi.X1','Roi.Y1','Roi.X2','Roi.Y2','ClassId'])

        return dataFrame

#Main
image_list = read_image(base_path="")
print("Length of image ", len(image_list))


