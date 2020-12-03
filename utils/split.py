from os import walk
import shutil
import os

current = "bald"
mypath = "dataset/"+current
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)

train = int(len(f) * .70)
valid = int(len(f) * .85)

os.mkdir("dataset/"+current+"-train")
os.mkdir("dataset/"+current+"-valid")
os.mkdir("dataset/"+current+"-test")
for i in range(train):
    shutil.copy("dataset/"+current+"/"+f[i], "dataset/"+current+"-train")

for i in range(train, valid):
    shutil.copy("dataset/"+current+"/"+f[i], "dataset/"+current+"-valid")

for i in range(valid, len(f)):
    shutil.copy("dataset/"+current+"/"+f[i], "dataset/"+current+"-test")
