from PIL import Image
import os

ON_WINDOWS = False
slash = '\\' if ON_WINDOWS else '/'

path = '..' + slash + 'dataset' + slash
bird_dirs = [x for x in os.listdir(path) if os.path.isdir(path+x)]
# print(bird_dirs)
extensions = {}


def resize_img():
    for bird in bird_dirs:
        print(bird + '...')
        for item in os.listdir(path+bird):
            file_ = path + bird + slash + item
            if os.path.isfile(file_):
                filename, extension = os.path.splitext(file_)
                if extension not in extensions:
                    extensions[extension] = 1
                else:
                    extensions[extension] += 1
                if extension.lower() == '.jpg' or extension.lower() == '.jpeg':
                    # print(filename)
                    image = Image.open(file_)
                    if image.size != (256, 256):
                        resized = image.resize((256, 256), Image.ANTIALIAS)
                        resized.save(filename + extension, 'JPEG', quality=90)
                elif extension == '.png':
                    image = Image.open(file_)
                    if image.size != (256, 256):
                        resized = image.resize((256, 256), Image.ANTIALIAS)
                        resized.save(filename + extension, 'png', quality=90)
                # if extension == '' or filename[-1] == '.':
                #   print('REMOVED', filename)
                #   os.remove(filename+extension)
    print(extensions)


resize_img()
