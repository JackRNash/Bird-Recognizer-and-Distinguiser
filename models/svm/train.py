import numpy
import os
import cv2
from sklearn import preprocessing
from sklearn.svm import SVC

bird_types = ["bald_eagle", "barn_owl", "belted_kingfisher", "blue_jay",
              "chipping_sparrow", "osprey", "red_bellied_woodpecker",
              "red_tailed_hawk", "red_winged_blackbird", "tree_swallow"]
image_path = "../../dataset/"


def label_and_rgb_images(stage):
    """
    Returns the raw BGR pixel data & correct labelings for each data point.

    Extract image nparrays & their matching classification from file system.

    Parameter stage: the current state of model creation process
    Precondition: stage is one of train, validation, test
    """
    X_train_raw = []
    y_train = []
    classif = 0

    for b_type in bird_types:
        files = os.listdir(image_path+stage+"/"+b_type)
        # Extract pixel data for each image in file system
        for f in files:
            if f != ".DS_Store":
                X_train_raw.append(cv2.imread(
                    image_path+stage+"/"+b_type+"/"+f))
                y_train.append(classif)
        classif += 1
    return X_train_raw, y_train


def extract_features(x_tr_raw):
    """
    Returns the feature vector representation of each raw BGR pixel data point.

    Extract feature vectors for each image in x_tr_raw.

    Parameter x_tr_raw: raw BGR pixel data
    Precondition: x_tr_raw is a 3D numpy array of rows containing BGR pixel
      objects
    """
    x_tr = []
    for img in x_tr_raw:
        features = []
        # Get mean & standard deviation of entire image BGR object
        mean, std = cv2.meanStdDev(img)

        # Add average & standard entire R value to features
        features.append(mean[2][0])
        features.append(std[2][0])
        # Add average & standard entire G value to features
        features.append(mean[1][0])
        features.append(std[1][0])
        # Add average & standard entire B value to features
        features.append(mean[0][0])
        features.append(std[0][0])

        x_tr.append(features)

    return x_tr


def preprocess_data(X, scalar="none"):
    """
    Returns scalar value used to normalize the data and the normalized data.

    Normalize data according to the scalar that makes the training features
    have a mean of zero and standard deviation of one.

    Parameter X: the feature data to normalize
    Precondition: X is a list of feature data

    Parameter scalar: the scalar by which to normalize the data
    Precondition: scalar is a StandardScalar SKLearn object or "none"
    """
    # Only generate a scalar if we are normalizing the training data
    if scalar == "none":
        scalar = preprocessing.StandardScaler().fit(X)
    return scalar, scalar.transform(X)


def generate_svm(X, y, c):
    """
    Returns SVM model.

    Creates SVM model using X and y as training data and labels respectively.

    Parameter X: feature vectors used for training
    Precondition: X is a 2D array containing feature vectors

    Parameter y: labels for each feature vector
    Precondition: y is a list containing the correct label for each feature 
      vector
    """
    svm = SVC(C=c)
    svm.fit(X, y)
    return svm


def calculate_error(preds, acc):
    """
    Returns percentage of incorrect labelings.

    Calculates error of predictions compared to actual values.

    Parameter preds: predictions for each image
    Precondition: preds is a list

    Parameter acc: actual labels for each image
    Precondition: acc is a list
    """
    assert len(preds) == len(acc), "Prediction & Actual must be same length"

    error = 0
    for i in range(len(preds)):
        if preds[i] != acc[i]:
            error += 1
    return error/len(preds)


if __name__ == "__main__":

    nums = ["1", "10", "100", "250", "500", "1000", "1500", "2000", "2500",
            "3000", "3500", "4000", "4500", "5000", "5500", "6000",
            "6500", "7000", "7500", "8000", "8500", "9000", "9500", "10000"]
    train_errors = []
    val_errors = []
    for n in nums:

        # Process training data and generate model
        X_tr_raw, y_tr = label_and_rgb_images("train")
        X_tr = extract_features(X_tr_raw)
        scalar, X_tr = preprocess_data(X_tr)
        svm_model = generate_svm(X_tr, y_tr, int(n))

        # Use validation set
        X_valid_raw, y_valid = label_and_rgb_images("validation")
        X_valid = extract_features(X_valid_raw)
        _, X_valid = preprocess_data(X_valid, scalar)
        preds = svm_model.predict(X_valid)
        val_errors.append(calculate_error(preds, y_valid))
        preds = svm_model.predict(X_tr)
        train_errors.append(calculate_error(preds, y_tr))
    print(nums)
    print(train_errors)
    print(val_errors)
