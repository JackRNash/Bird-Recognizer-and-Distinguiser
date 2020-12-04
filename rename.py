import os
import shutil
from os import listdir
from os.path import isfile, join

current = "bald"
mypath = "/Users/connoranderson/Desktop/Bird-Recognizer-and-Distinguisher/dataset/"+current

f = [files for files in listdir(mypath) if isfile(join(mypath, files))]

for i in range(1, len(f)):
    ending = f[i].split(".")
    os.rename(mypath+"/"+f[i],
              mypath + "/" + str(i)+"."+ending[1])
