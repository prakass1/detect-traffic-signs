import cv2
import matplotlib.pyplot as plt
import  numpy as np
import csv

base_path = "F://Modules_Courses//Semester-4//Computer Vision//data//GTSRB//Final_Training//Images//"
ext = ".csv"
file_cnst = "GT-"

dict_img = {}
for i in range(0,43):

    if len(str(i)) == 1:
        prefix = str(i).zfill(5)
    elif len(str(i)) == 2:
        prefix = str(i).zfill(5)

    reader = open(base_path + prefix + "//" + "".join(file_cnst + prefix + ext))

    csv_reader = csv.reader(reader, delimiter=";")
    next(csv_reader)

    row_list = []

    for row in csv_reader:
        print(base_path + prefix + "//" + row[0])
        row_list.append(row[0])

    img_arr = cv2.imread(base_path + prefix + "//" + row_list[-1])
    img_resized = cv2.resize(img_arr,(32,32))
    dict_img[i] = img_resized


count = 0
fig, ax = plt.subplots(7, 7, figsize=(15, 10))
plt.title("Dataset Images")
for i in range(6):
    for j in range(7):
        ax[i][j].imshow(dict_img.get(count))
        ax[i][j].set_xlabel(str(count))
        count = count + 1
#Manually append since it is odd
ax[6][0].imshow(dict_img.get(42))
ax[6][0].set_xlabel(str(42))
plt.tight_layout()
plt.savefig("image_ds.png")

