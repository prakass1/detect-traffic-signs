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
Jannes Redler (212923)

Running System:
Python Version: 3.X
OS: Windows/Linux

'''

import props as properties
import machine_learning as ml
import argparse
import matplotlib.pyplot as plt
import sys
import cv2
import os

from collections import Counter
import template_matching_color_scale as tm

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
        print("Started Single image prediction")
        if type(args.filename) is not str:
            print("please pass the filename with its path to the --filename argument")
            sys.exit(1)
        else:
            # Calling only hog predictions because voting seems not good

            predictions = []
            #img_arr = cv2.imread(args.filename)
            localized_img = tm.do_tm(img_loc=args.filename, temp_loc="templates//*.png")
            for feature in ["hog"]:
                predictions.append(ml.make_single_img_prediction(feature, localized_img))
                #class_votes = Counter(predictions)
            pred = predictions[0]
            #for key, value in class_votes.items():
                #if value > 1:
                    #pred = key
                    #break

            pred_class = ml.class_switcher(str(pred[0]))
        
            print("The prediction for the image is :" + str(pred[0]) + " - " + pred_class)
            plt.imshow(cv2.cvtColor(localized_img,cv2.COLOR_BGR2RGB))
            plt.text(0.5, 0.5, str(pred[0]) + " - " + pred_class, horizontalalignment='left', verticalalignment='top', color="g", weight="bold")
            pred_img = "images//prediction.png"
            print("Saving the prediction image at - " + str(pred_img))
            plt.savefig(pred_img)
            plt.close()
            return
        

    if properties.train is True:
        for feature in ["hog"]: #"gray","hsv","laplacian","merged"]:
            print("Started Training")
            ml.perform_training("rf", features=feature)

    if properties.predict is True:
        for feature in ["hog"]:
            print("Started testing")
            ml.make_predict(features=feature)
    return

def initialize_args():
    
    global args
    args = parser.parse_args()

    import os
    if not os.path.isdir("predictions"):
        print("Creating predictions directory")
        os.makedirs("predictions")
    if not os.path.isdir("images"):
        print("Creating images directory")
        os.makedirs("images")

    if type(args.train) is bool:
        print("Setting {} for train ".format(str(args.train)))
        properties.train = args.train
    
    if type(args.predict) is bool:
        print("Setting {} for predict ".format(str(args.predict)))
        properties.predict = args.predict

    if type(args.model_location) is str:
        print("Setting {} for model location ".format(str(args.model_location)))
        properties.model_location = args.model_location

    if type(args.train_base_dir) is str:
        print("Setting {} for train base directory ".format(str(args.train_base_dir)))
        properties.train_base_dir = args.train_base_dir

    if type(args.test_base_dir) is str:
        print("Setting {} for test base directory".format(str(args.train)))
        properties.test_base_dir = args.test_base_dir


    

#### Start of Main ####


if __name__ == '__main__':
    main()

