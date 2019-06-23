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
Oliver Watson (224262)
Jannes Randler (212923)

Running System:
Python Version: 3.X
OS: Windows/Linux

'''

import props as properties
import machine_learning as ml
import argparse
import matplotlib.pyplot as plt
import cv2

from collections import Counter


# bool is not handled well using argparse. wrapper to convert str to bool value
def str_to_bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


parser = argparse.ArgumentParser(description='')
parser.add_argument('--train', type=str_to_bool, help='training flag - indicates if training is to performed')
parser.add_argument('--predict', type=str_to_bool, help='predict flag - indicates if prediction is to be performed')
parser.add_argument('--model_location', type=str, help='directory to save the model')
parser.add_argument('--train_base_dir', type=str, help='directory for the training data')
parser.add_argument('--test_base_dir', type=str, help='directory for the test data')
parser.add_argument('--single', type=str_to_bool, help='classify a single full scene image')
parser.add_argument('--filename', type=str, help='path to file to be classified')

args = None

def main():

    initialize_args()

    if type(args.single) is bool:
        if type(args.filename) is not str:
            print("please pass the filename with its path to the --filename argument")
        else:
            # uses majority voting to determine class, if no clear winner use prediction from hog model
            predictions = []
            img_arr = cv2.imread(args.filename)
            for feature in ["merged"]:
                predictions.append(ml.make_single_img_prediction(feature, img_arr))

            class_votes = Counter(predictions)

            pred = predictions[0]

            for key,value in class_votes.items(): 
                if value > 1:
                    pred = key
                    break
            
            pred_class = ml.class_switcher(pred)
        
            print("The prediction for the image is :" + pred + " - " + pred_class)
            plt.imshow(args.filename, cmap="gray")
            plt.text(0.5, 0.5, pred + " - " + pred_class, horizontalalignment='left', verticalalignment='top', color="g", weight="bold")
            plt.savefig("images//prediction.png")
            return
        


    if properties.train == True:
        for feature in [ "hog"]: #,"gray","hsv","laplacian","merged"]:
            ml.perform_training("rf", features=feature)

    if properties.predict == True:
        for feature in ["merged"]:
            ml.make_predict(features=feature)

def initialize_args():
    
    global args
    args = parser.parse_args()

    if type(args.train) is bool:
        properties.train = args.train
    
    if type(args.predict) is bool:
        properties.predict = args.predict

    if type(args.model_location) is str:
        properties.model_location = args.model_location

    if type(args.train_base_dir) is str:
        properties.train_base_dir = args.train_base_dir

    if type(args.test_base_dir) is str:
        properties.test_base_dir = args.test_base_dir


    

#### Start of Main ####


if __name__ == '__main__':
    main()
