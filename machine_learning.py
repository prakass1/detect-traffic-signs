from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import *
import props as properties
import processing as pp
import pickle
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def build_model(X, classId, model_name, features):
    try:
        #X_train, X_test, y_train, y_test = train_test_split(X,
         #                                               classId,
          #                                              test_size = 0.33,
           #                                             random_state = np.random.randint(100,1001))
        if model_name == "rf":

            # The parameters are only obtained after grid search which is commented below
            rf = RandomForestClassifier(n_estimators=700, criterion="entropy")
            # Repeated kfolds with 2 splits and repeation of 5 times is done
            splits = 2
            repeats = 5
            rkf = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=None)
            count = 1
            for train_index, test_index in rkf.split(X):
                print("Starting training with fold - %d" % count)
                X_train, X_test = np.asarray(X)[train_index], np.asarray(X)[test_index]
                y_train, y_test = np.asarray(classId)[train_index], np.asarray(classId)[test_index]
                y_labels = np.unique(y_train)
                rf.fit(X_train, y_train)
                pred = rf.predict(X_test)
                pred_prob = rf.predict_proba(X_test)
                from pprint import pprint
                pprint(classification_report(pred, y_test))
                print("Precision Score - Micro Averaging is -- %f" % precision_score(pred, y_test, average="micro"))
                print("Recall Score - Micro Averaging is -- %f " % recall_score(pred, y_test, average="micro"))
                print("F1 Score - is -- %f" % f1_score(pred, y_test, average="micro"))
                print("Accuracy of fold- " + str(count) + " is -- %f" % accuracy_score(pred, y_test))
                cf_m = confusion_matrix(pred, y_test)
                pprint(cf_m)
                if count == (splits * repeats):

                    sns.set_style("whitegrid")
                    sns.set_palette("viridis")
                    cm_df = pd.DataFrame(cf_m, index=[class_switcher(i) for i in y_labels],
                                         columns=[class_switcher(i) for i in y_labels])
                    plt.figure(figsize=(18, 15))
                    try:
                        heatmap = sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGn")
                    except ValueError:
                        raise ValueError("Confusion matrix values must be integers.")
                    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',
                                                 fontsize=10)
                    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',
                                                 fontsize=10)
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')

                    if features is not None:
                        plt.savefig("images//cm-fold-"+ str(count) + str(features) + ".png")
                    else:
                        plt.savefig("images//cm-fold-" + str(count) + ".png")

                count += 1

    #Grid Search is now commented
        '''
        params = {
            # Number of trees
            "n_estimators": [400, 500, 700],
            "criterion": ["gini", "entropy"]
        }

        for score in ["precision", "recall"]:
            grid_cv = GridSearchCV(rf, param_grid=params, cv=5, n_jobs= -1, scoring="%s_macro" % score)
            grid_cv.fit(X_train, y_train)

            from pprint import pprint
            pprint(grid_cv.cv_results_)
            pprint(grid_cv.best_estimator_)
            pprint(grid_cv.best_params_)
            pprint(grid_cv.best_score_)
            means = grid_cv.cv_results_['mean_test_score']
            stds = grid_cv.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, grid_cv.cv_results_['params']):
                print("The mean score is ", mean)
                print("The std score is ", std)
                print("The param is ", params)

            # Make predictions
            pred = grid_cv.predict(X_test)

            #Classification report
            pprint(confusion_matrix(y_test, pred))
            pprint(accuracy_score(y_test, pred))
        '''
        # Some plots


        # Save the built model
        if features is not None:
            save_model("".join(model_name + "_" + features), rf)
        else:
            save_model("".join(model_name), rf)

    except Exception:
        print("There seems to be some error during model build phase")
        traceback.print_exc()
        return -1
    return 0


def perform_training(model_name, features=None):
    """
    Default features are none. This means that the image list is just normalized, flatten and returned to build
    machine learning models

    Options for features:
    1. gray - Grayscale
    2. hsv - Hue, Saturation, Value is extracted
    3. SIFT - Scale Invariant Feature Transformation
    4. HoG - Histogram of Oriented Gradients

    :param features:
    :return:
    """
    print("Training has started...")
    image_list, class_labels = pp.read_image(base_path=properties.train_base_dir)
    print("Length of image ", len(image_list))
    print("Extracting features from the read images...")
    X = pp.extract_features(features, image_list)
    # Build a machine learning model
    print("Machine learning module started for %s model with %s features" % (model_name, features))
    rc = build_model(X, class_labels, model_name, features)

    if rc == 0:
        print("Training and validation of model is now completed successfully!!")
    else:
        print("There is some error while building the model. Fix errors and retrain")


def make_predict(features):
    '''

    Steps:
    1. Read the built model from the provided location of properties
    2. Read the image from the image directory (1)
    3. Make prediction to provide results

    If image is a scene of real time data:
    1. Do scale invariant template matching and get the found confident region
    2. Extract the region with a bit of heuristics
    3. Use that image to Make prediction and draw a boundary box with prediction to the scene
    :return:
    '''

    image_list = pp.read_test_image(properties.test_base_dir, False)
    X = pp.extract_features(features, image_list)
    

    print("Starting Testing...")
    model = pickle.load(open(properties.model_location + "rf_" + features, 'rb'))
    pred = model.predict(X)
    pred_prob = model.predict_proba(X)
    print("Testing complete...")

    predict_arr = []

    for i, row in enumerate(pred_prob):
        new_list = []
        
        for val in row:
            new_list.append(val)

        if max(row) > 0.8:
            new_list.append(class_switcher(pred[i]))
        else:
            new_list.append("Others")
        
        predict_arr.append(new_list)


    ### Write prediction to test files
    np.savetxt("predictions/" + features + "/prediction.csv", np.array(pred), fmt='%s', delimiter=",")
    
    np.savetxt("predictions/" + features + "/prediction_prob.csv", np.array(pred_prob), delimiter=",")

    np.savetxt("predictions/" + features + "/prediction_all.csv", np.array(predict_arr), fmt='%s', delimiter=",")
    
def class_switcher(arg):
    switch = {
        '3': 'Speed Sign',
        '11': 'Priority to through-traffic',
        '12': 'Priority Road starts',
        '13': 'Yield'
    }

    return switch.get(arg, "Others")


def save_model(model_name, obj):
    print("Saving model to the location ", properties.model_location)
    model_file = open("".join(properties.model_location + model_name), "wb")
    # Save the model
    pickle.dump(obj, model_file)