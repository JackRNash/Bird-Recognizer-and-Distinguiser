import shutil
from os import listdir
import os
from os.path import isfile, join

current = "woodpecker"
mypath = "/Users/connoranderson/Desktop/Bird-Recognizer-and-Distinguisher/dataset/"+current
f = []
f = [files for files in listdir(mypath) if isfile(join(mypath, files))]
print(f)
train = int(len(f) * .70)
valid = int(len(f) * .85)

os.mkdir("dataset/"+current+"-train")
os.mkdir("dataset/"+current+"-valid")
os.mkdir("dataset/"+current+"-test")
for i in range(train):
    print("Copying: " + "dataset/"+current+"/"+f[i])
    shutil.copy("dataset/"+current+"/"+f[i], "dataset/"+current+"-train")

for i in range(train, valid):
    shutil.copy("dataset/"+current+"/"+f[i], "dataset/"+current+"-valid")

for i in range(valid, len(f)):
    shutil.copy("dataset/"+current+"/"+f[i], "dataset/"+current+"-test")
