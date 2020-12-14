import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.svm import SVC

bird_types = ["bald_eagle", "barn_owl", "belted_kingfisher", "blue_jay",
              "chipping_sparrow", "osprey", "red_bellied_woodpecker",
              "red_tailed_hawk", "red_winged_blackbird", "tree_swallow"]
image_path = "../../dataset/"


def extract_center(img):
    """
    Returns the raw BGR pixels for the center 128x128 of an image.

    Extracts inner pixels of an image.

    Parameter img: the entire raw BGR pixels of an image
    Precondition: it is a 256x256 image
    """
    center = []
    for i in range(64, 192):
        center.append(img[i][64:192])
    return center


def label_and_rgb_images(stage):
    """
    Returns the raw BGR pixel data & correct labelings for each data point.

    Extract image nparrays & their matching classification from file system.

    Parameter stage: the current state of model creation process
    Precondition: stage is one of train, validation, test
    """
    X_train_raw = []
    X_train_raw_center = []
    y_train = []
    classif = 0

    for b_type in bird_types:
        files = os.listdir(image_path+stage+"/"+b_type)
        # Extract pixel data for each image in file system
        for f in files:
            if f != ".DS_Store":
                img = cv2.imread(image_path+stage+"/"+b_type+"/"+f)
                X_train_raw.append(img)
                X_train_raw_center.append(extract_center(img))
                y_train.append(classif)
        classif += 1
    return X_train_raw, X_train_raw_center, y_train


def extract_features(x_tr_raw, x_tr_raw_c):
    """
    Returns the feature vector representation of each raw BGR pixel data point.

    Extract feature vectors for each image in x_tr_raw.

    Parameter x_tr_raw: raw BGR pixel data
    Precondition: x_tr_raw is a 3D numpy array of rows containing BGR pixel
      objects
    """
    assert len(x_tr_raw) == len(x_tr_raw_c), "Inputs must be of same length"
    assert type(x_tr_raw) == type(x_tr_raw_c), "Inputs must be of same type"

    x_tr = []
    for i in range(len(x_tr_raw)):
        features = []
        # Get mean & standard deviation of entire image BGR object
        mean_full, std_full = cv2.meanStdDev(x_tr_raw[i])
        mean_cent, std_cent = cv2.meanStdDev(np.float32(x_tr_raw_c[i]))

        # Add average & standard entire R value to features
        features.append(mean_full[2][0])
        features.append(std_full[2][0])
        # Add average & standard entire G value to features
        features.append(mean_full[1][0])
        features.append(std_full[1][0])
        # Add average & standard entire B value to features
        features.append(mean_full[0][0])
        features.append(std_full[0][0])

        # Add average & standard center R value to features
        features.append(mean_cent[2][0])
        features.append(std_cent[2][0])
        # Add average & standard center G value to features
        features.append(mean_cent[1][0])
        features.append(std_cent[1][0])
        # Add average & standard center B value to features
        features.append(mean_cent[0][0])
        features.append(std_cent[0][0])

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


def generate_svm(X, y, c=1000):
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


def plot_confusion_matrix(true, preds, classes):
    assert len(true) == len(preds), "Oh no"
    cmatrix = confusion_matrix(true, preds)
    threshold = np.min(np.diagonal(cmatrix))
    fig, ax = plt.subplots()
    ax.set(xticks=np.arange(cmatrix.shape[1]),
           yticks=np.arange(cmatrix.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    ax.tick_params(axis='x', labelrotation=90)
    for i in range(cmatrix.shape[0]):
        for j in range(cmatrix.shape[1]):
            if cmatrix[i, j] > 0:
                ax.text(j, i, cmatrix[i, j], ha='center', va='center',
                        color='white' if cmatrix[i, j] < threshold else 'black')
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    ax.imshow(cmatrix)


if __name__ == "__main__":
    # C = 38 is best C value
    nums = []
    curr = 38
    for i in range(1):
        nums.append(curr)
        curr += 1
    train_errors = []
    val_errors = []
    for n in nums:

        # Process training data and generate model
        X_tr_raw, X_tr_raw_c, y_tr = label_and_rgb_images("train")
        X_tr = extract_features(X_tr_raw, X_tr_raw_c)
        scalar, X_tr = preprocess_data(X_tr)
        svm_model = generate_svm(X_tr, y_tr, n)

        # Use validation set
        X_valid_raw, X_valid_raw_c, y_valid = label_and_rgb_images(
            "validation")
        X_valid = extract_features(X_valid_raw, X_valid_raw_c)
        _, X_valid = preprocess_data(X_valid, scalar)
        preds_val = svm_model.predict(X_valid)
        print(1-calculate_error(preds_val, y_valid))
        preds = svm_model.predict(X_tr)
        print(1-calculate_error(preds, y_tr))

        X_test_raw, X_test_raw_c, y_test = label_and_rgb_images(
            "test")
        X_test = extract_features(X_test_raw, X_test_raw_c)
        _, X_test = preprocess_data(X_test, scalar)
        preds_test = svm_model.predict(X_test)
        print(1-calculate_error(preds_test, y_test))
    # plot_confusion_matrix(y_valid, preds_val, ['bald_eagle', 'barn_owl', 'belted_kingfisher', 'blue_jay', 'chipping_sparrow',
    #                                            'osprey', 'red_bellied_woodpecker', 'red_tailed_hawk', 'red_winged_blackbird', 'tree_swallow'])
    # plt.savefig('svm_conf.png')
    # plt.plot(nums, [x*100 for x in train_errors], 'r', label="training")
    # plt.plot(nums, [x*100 for x in val_errors], 'b', label="validation")
    # plt.axis([0, 500, 0, 100])
    # plt.xlabel("C value")
    # plt.ylabel("Percent error")
    # plt.title("Error vs C value for SVM w/ 12 general features")
    # plt.legend(loc="upper right")
    # plt.savefig('12_feature.png')
    # plt.clf()
