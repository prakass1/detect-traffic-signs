'''
This class will ideally do initial bit to extract the csvs of the required classes
and extract key features to create a data to feed into machine learning models

Authors:


Running system:
python 3.6

'''

import props as properties
import machine_learning as ml

#switch to args later
def main():

    if properties.args == "cl":
        initialize_args()

    if properties.train == True:
        ml.perform_training("rf", features="hog")

    if properties.predict == True:
        ml.make_predict(features="hog")


def initialize_args():
    
    #add reading of args from command line

    return

#### Start of Main ####


if __name__ == '__main__':
    main()
