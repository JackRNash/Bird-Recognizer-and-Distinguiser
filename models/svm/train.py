import numpy
import os
from os import listdir
from os.path import isfile, join
from PIL import Image

bird_types = ["bald_eagle", "barn_owl", "belted_kingfisher", "blue_jay",
              "chipping_sparrow", "osprey", "red_bellied_woodpecker",
              "red_tailed_hawk", "red_winged_blackbird", "tree_swallow"]

mypath = "/Users/connoranderson/Desktop/Bird-Recognizer-and-Distinguisher/dataset/"

X_train = []
y_train = []

for b_type in bird_types:
    pass
f = [files for files in os.listdir(mypath) if isfile(join(mypath, files))]
