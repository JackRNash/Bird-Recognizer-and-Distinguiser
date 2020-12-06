#!/usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn.svm import SVC
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import time
import pickle
import argparse


def process(arr):
    samples = arr.shape[-1]
    features = np.zeros((samples, 8))
    for i in range(samples):
        hsv = colors.rgb_to_hsv(arr[:, :, :3, i])
        np.mean(hsv, axis=(0, 1), out=features[i, :3])
        np.std(hsv, axis=(0, 1), out=features[i, 3:6])
        features[i, 6] = np.mean(arr[:, :, :, 3])
        features[i, 7] = np.std(arr[:, :, :, 3])
    scaler = MinMaxScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    return features


def split(X, Y, size):
    start = time.time()
    X, Y = shuffle(X.T, Y.T, random_state=2720)
    X, Y = X.T, Y.T
    xTr = process(X[:, :, :, :size])
    yTr = Y[:size]
    xTe = process(X[:, :, :, size:])
    yTe = Y[size:]
    print 'Processing & splitting the data took {:.2f}s'.format(time.time()-start)
    print 'Size of training data: {} samples'.format(xTr.shape[0])
    print 'Size of test data: {} samples'.format(xTe.shape[0])
    return xTr, yTr, xTe, yTe


def train(X, Y, kernel='linear', gamma='auto', C=1, verbose=True):
    start = time.time()
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=2720)
    model.fit(X, Y)
    if verbose:
        print 'Training took {:.2} seconds'.format(time.time() - start)
    return model


def eval(X, Y, model, display_error=False):
    preds = model.predict(X)
    acc = len(preds[preds == Y])/len(Y)
    if display_error:
        L = ['barren land', 'trees', 'grassland', 'none']  # labels
        filter = preds != Y
        errors = preds[filter]
        true_of_errors = Y[filter]

        locs = np.nonzero(filter)
        fig, axs = plt.subplots(2, 2)
        for i in range(2):
            for j in range(2):
                num = 2*i + j
                img = test_x[:, :, :, locs[0][num]]
                axs[i, j].imshow(img[:, :, :3])
                axs[i, j].set_title('Predicted: {}\n Actual: {}'.format(
                          L[errors[num]], L[true_of_errors[num]])
                       )
                # https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot
                axs[i, j].axis('off')
        st = fig.suptitle('Misclassified Images', fontsize="x-large")
        fig.tight_layout()
        st.set_y(0.95)
        fig.subplots_adjust(top=0.80)
        plt.savefig('misclassified', bbox_inches='tight')
        plt.clf()

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
        con_mat = confusion_matrix(Y, preds)
        fig, ax = plt.subplots()
        ax.imshow(con_mat)

        ax.set(xticks=np.arange(con_mat.shape[1]),
               yticks=np.arange(con_mat.shape[0]),

               xticklabels=L, yticklabels=L,
               title="Confusion Matrix",
               ylabel='True label',
               xlabel='Predicted label')

        for i in range(con_mat.shape[0]):
            for j in range(con_mat.shape[1]):
                ax.text(j, i, con_mat[i, j],
                        ha="center", va="center",
                        color="white" if con_mat[i, j] < 30000 else "black")
        fig.tight_layout()
        plt.savefig('confusion_matrix', bbox_inches='tight')
    return acc


def train_eval(xTr, yTr, xTe, yTe, kernel='linear',
               gamma='auto', C=1, verbose=True):
    model = train(xTr, yTr, kernel=kernel, gamma=gamma, C=C)
    start = time.time()
    train_acc = eval(xTr, yTr, model)
    if verbose:
        print 'Training Accuracy: {:.2f}% ({:.2f}s)'.format(train_acc*100,
                                                            time.time()-start)
    start = time.time()
    test_acc = eval(xTe, yTe, model)
    if verbose:
        print 'Test Accuracy: {:.2f}% ({:.2f}s)'.format(test_acc*100,
                                                        time.time()-start)
    return model


def param_search(xTr, yTr, xTe, yTe, kernel='rbf', Cs=[.1, 1, 10, 100, 1000],
                 gammas=['auto', 1, 10, 100]):
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    accs = np.zeros((len(Cs), len(gammas)))
    start = time.time()
    print 'Conducting a hyperparameter search...'
    for i, C in enumerate(Cs):
        for j, g in enumerate(gammas):
            model = train(xTr, yTr, kernel=kernel, gamma=g, C=C, verbose=False)
            accs[i, j] = eval(xTe, yTe, model)
    print 'Took {:.2f}s to train all {} models'.format(
           time.time()-start, accs.shape[0]*accs.shape[1]
    )
    c_index, g_index = np.unravel_index(accs.argmax(), accs.shape)
    print 'Optimal hyperparameters are C={} and gamma={}'.format(
          Cs[c_index], gammas[g_index]
    )
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(gammas)))
    ax.set_yticks(np.arange(len(Cs)))
    ax.set_xticklabels(gammas)
    ax.set_yticklabels(Cs)
    plt.setp(ax.get_xticklabels(), rotation='vertical', ha="left")
    #          rotation_mode="anchor")
    ax.imshow(accs)
    ax.set_ylabel('Error penalty')
    ax.set_xlabel('Kernel Coefficient')
    ax.set_title('Test Accuracy Across Different Hyperparameter Values')
    for i in range(accs.shape[0]):
        for j in range(accs.shape[1]):
            text = ax.text(j, i, '{:.3f}'.format(accs[i, j]),
                           ha="center", va="center", color="black")
    fig.tight_layout()
    plt.savefig('gridsearch', bbox_inches='tight')
    return accs


if __name__ == "__main__":
    DIR = '/classes/ece2720/fpp/'
    # DIR = ''
    MODEL_NAME = 'model.dat'
    parser = argparse.ArgumentParser(description="Train satellite image classification model")
    parser.add_argument("--load", action='store_true', help='Load model instead of training, creates confusion_matrix & misclassified graphics')
    parser.add_argument("--paramsearch", action='store_true', help='Conduct a parameter search across a medium sized dataset')
    args = parser.parse_args()
    LOAD_MODE = args.load  # Load model instead of training, creates confusion_matrix & misclassified graphics
    MEDIUM_PARAM_SEARCH = args.paramsearch  # Conduct a parameter search across a medium sized dataset

    print '-'*90
    print ' '*15, "Jack Nash's final programming project for ECE 2720 - Training"
    print '-'*90

    start = time.time()
    mat = loadmat(DIR+'sat-4-full.mat')
    print 'Dataset loaded ({:.2f}s)'.format(time.time()-start)

    train_x = mat['train_x']
    train_y = mat['train_y']
    test_x = mat['test_x']
    test_y = mat['test_y']

    # Undo one-hot-encoding
    train_y = np.argmax(train_y, axis=0)
    test_y = np.argmax(test_y, axis=0)

    if MEDIUM_PARAM_SEARCH:
        medium_x = train_x[:, :, :, :10000]
        medium_y = train_y[:10000]

        xTr, yTr, xTe, yTe = split(medium_x, medium_y, 7000)
        param_search(xTr, yTr, xTe, yTe, kernel='rbf',
                     Cs=[.1, 1, 5, 10, 50, 100, 1000],
                     gammas=['auto', 1, 10, 30, 50, 100])

    if LOAD_MODE:
        print 'In LOAD MODE, not training a new model. Did you intend this?'
        model = pickle.load(open('model.dat', 'rb'))
        display = True
    else:
        xTr, yTr, xTe, yTe = split(train_x, train_y, 300000)
        model = train_eval(xTr, yTr, xTe, yTe, kernel='rbf', gamma=10, C=30)
        display = False

    start = time.time()
    xVal = process(test_x)
    print 'Processed validation set ({})'.format(time.time() - start)

    start = time.time()
    print 'Validation accuracy: {:.2f}% ({})'.format(
           eval(xVal, test_y, model, display_error=display)*100,
           time.time() - start
       )

    if not LOAD_MODE:
        print "Saving model, don't interrupt..."
        pickle.dump(model, open(MODEL_NAME, 'wb'))
        print 'Model saved as', MODEL_NAME
