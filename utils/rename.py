import os
import shutil
from os import listdir
from os.path import isfile, join

current = "woodpecker"
mypath = "/Users/connoranderson/Desktop/Bird-Recognizer-and-Distinguisher/dataset/"+current

f = [files for files in listdir(mypath) if isfile(join(mypath, files))]

for i in range(0, len(f)):
    ending = f[i].split(".")
    os.rename(mypath+"/"+f[i],
              mypath + "/" + str(i+1)+"."+ending[1])
