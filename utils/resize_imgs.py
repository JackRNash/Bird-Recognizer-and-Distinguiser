from PIL import Image
import os

path = '..\\dataset\\'
bird_dirs = [x for x in os.listdir(path) if os.path.isdir(path+x)]
print(bird_dirs)

def resize_img():
  for bird in bird_dirs:
    for item in os.listdir(path+bird):
      file_ = path + bird + '\\' + item
      if os.path.isfile(file_):
        filename, extension = os.path.splitext(file_)
        if extension == '.jpg':
          print(filename)
          image = Image.open(file_)
          resized = image.resize((256, 256), Image.ANTIALIAS)
          resized.save(filename+'.jpg', 'JPEG', quality=90)
        # elif extension == '':
        #   print('REMOVED', filename)
        #   os.remove(filename)

resize_img()