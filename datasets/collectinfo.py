from pprint import pprint
import os
import json
import argparse

parser = argparse.ArgumentParser(description='dataset')
parser.add_argument('--classes', default='yourVideo')
opt = parser.parse_args()

folder1 = "./DTexture/processed/" + opt.classes + "/"

with open("./DTexture/processed/" + opt.classes + "ID.txt", "w") as text_file:
    for foldername in os.listdir(folder1): 
        print(foldername)
        subfolder = folder1 + foldername + "/"
        path, dirs, files = os.walk(subfolder).next()
        file_count = len(files)
        text_file.write(foldername)
        text_file.write(" 1 ")
        text_file.write('%d' % file_count)
        text_file.write("\n")

