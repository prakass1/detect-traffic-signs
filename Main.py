'''
A wrapper to run all the functions
train
- During training run to read images for training,
- extract valuable feature descriptors,
- train ml model and save it.
- Plot learning curves

test
- Use the trained model
- Extract related model features
- Use it to make prediction over test images
- Plot a sample of predictions

Authors:
Subash Prakash (220408)
Oliver Watson
Jannes Randler

Running System:
Python Version: 3.X
OS: Windows/Linux

'''

import props as properties
import machine_learning as ml

#switch to args later
def main():

    if properties.args == "cl":
        initialize_args()

    if properties.train == True:
        for feature in ["gray", "hsv", "laplacian"]:
            ml.perform_training("rf", features=feature)

    if properties.predict == True:
        for feature in ["hog", "gray", "hsv", "laplacian"]:
            ml.make_predict(features=feature)

def initialize_args():
    
    #add reading of args from command line

    return

#### Start of Main ####


if __name__ == '__main__':
    main()
