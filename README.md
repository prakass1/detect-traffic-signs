# Setting up the Environment for running the code

## Initial Step:
Operating System: Windows

1. Go to [python installation!!](https://www.python.org/downloads/) and download the setup for python and install it. 
Once, that is setup ensure that the environment variable are all set for python by checking below command on windows command prompt:
    * Open command prompt
    * type **python --version** (This should return a version, if not the environment variables are not setup correctly.)
    * Setup pip next using this tutorial: [pip installation tutorial](https://pip.pypa.io/en/stable/installing/)
    * Now you are all ready with install all required libraries for running python machine learning code.

2. This repo also contains a requirements.txt using which one can install all the libraries on the fly at once.
    * cd into the extracted folder and run **pip install -r requirements.txt** . With all things in place this should setup all required libraries correctly

Now the system is ready for running the code.

## Alternative Step (Easy):
1. Installation of Anaconda:
Anaconda is a virtual environment which already comes with latest libraries for machine learning and usages.
Install Anaconda using this link : [anaconda installation](https://www.anaconda.com/)

2. Using the anaconda navigator GUI. All required libraries can be installed:

    *   pandas==0.24.2
    *   seaborn==0.9.0
    *   matplotlib==2.2.3 
    *   scikit-learn==0.19.1
    *   numpy==1.15.1
    *   opencv==3.4.2

If any libraries are missed, there would be error and will need to be installed.


## Steps for Execution of Python Code:
-- The python code is structured into below modules:
* Main.py - This module is a wrapper which runs to either train or predict over a machine learning model.
* props.py - This is the place where are configuration such as 
  * classes = [str(i) for i in range(0, 43)] (This sets the class labels)
  * train = True (This can be true or false meaning either to train the machine learning model or not)
  * predict = True (Same as train)
  * model_location = "" (location to store the models which are trained)
  * train_base_dir = "" (Base directory where the training images are resided)
  * test_base_dir = "" (Base directory where the test images are located)
*   machine_learning.py - This module contains methods to build a machine learning model. There are capabilities to extend to many models
*   processing.py - This module contains methods to perform feature extraction and reading of images into arrays etc...
    

-- Ideally, to run the system one would use below parameters in arguments:

Help:
*  python Main.py -h will provide the help for running the script

*_Note: Ensure that you add "/" at the end of each folder locations_*
   * Example:  
   
   <span style="color:green;">Correct usage: "F:/training_images/"</span>  
   <span style="color:red;">False usage: "F:/training_images"</span>

### Training a machine learning model and make prediction:
*  >python Main.py --train t --predict t --model_location models/ --train_base_dir "F:/GTSRB/Final_Training/Images/" --test_base_dir "F:/GTSRB/Final_Test/Images/"  

### Prediction of Test Images Only:
*  >python Main.py --predict t --model_location models/ --test_base_dir "F:/GTSRB/Final_Test/Images/"  

### Training of machine learning model Only:
*  >python Main.py --train t --model_location models/ --train_base_dir "F:/GTSRB/Final_Training/Images/"  

### Prediction on the Scene:
*  >python Main.py --single t --filename "templates/images/19.jpg"  
