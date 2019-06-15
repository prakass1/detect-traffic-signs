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
Jannes Randler

Running System:
Python Version: 3.X
OS: Windows/Linux

'''

import props as properties
import machine_learning as ml
import argparse


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


def main():

    initialize_args()

    if properties.train == True:
        for feature in ["hog","gray", "hsv", "laplacian", "merged"]:
            ml.perform_training("rf", features=feature)

    if properties.predict == True:
        for feature in ["hog", "gray", "hsv", "laplacian"]:
            ml.make_predict(features=feature)

def initialize_args():
    
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
