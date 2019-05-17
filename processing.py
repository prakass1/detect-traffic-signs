'''
This class will ideally do initial bit to extract the csvs of the required classes
and extract key features to create a data to feed into machine learning models

Authors:


Running system:
python 3.6

'''

import csv
import props
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
ext = ".csv"
file_const = "GT-"
def read_image(base_path=".",roi=False):
        image_list = []
        class_labels = []

        print("Image Extracting started !!")

        for val in props.classes:
            # Read the file by using the prefix and base path
            reader = open(base_path + val + "//" + "".join(file_const + val + ext))
            csv_reader = csv.reader(reader, delimiter=';')
            next(csv_reader)

            for row in csv_reader:
                im = cv2.imread(base_path + val + "//" + row[0],cv2.IMREAD_GRAYSCALE)
                #im_resized = cv2.resize(im,(32, 32))
                image_list.append(im)
                class_labels.append(row[7])
                #print("Image Filename - ", row[0])

            #close somewhere finally
            reader.close()
            print("Extraction is now completed for class -- " + str(val) + " and current images are -- " + str(len(image_list)))
        return image_list, class_labels


#Main
image_list, class_labels = read_image(base_path="")
print("Length of image ", len(image_list))

#Visualize an image
ima = image_list[6889]
plt.imshow(ima)

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

'''
#### normalization of images wiki
X = np.array(image_list) / 255
type(X[0])
X = [x - np.mean(x) for x in X]

X = [x.flatten() for x in X]
#####################################

#### Example to implement an ML - Surely overfitting #################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

X_train, X_test, y_train, y_test = train_test_split(X, class_labels, test_size=0.50, random_state=42)


rf = RandomForestClassifier()
rf.fit(X_train,y_train)
pred = rf.predict(X_test)

accuracy_score(y_test,pred)
confusion_matrix(y_test,pred)
'''
