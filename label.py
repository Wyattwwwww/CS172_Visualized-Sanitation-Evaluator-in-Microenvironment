import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def label(name, batch):
    data_path = name
    subdirs = os.listdir(data_path)
    files = {}
    exist = os.path.exists(data_path + "_label.txt")
    if exist:
        os.remove(data_path + "_label.txt")

    for subdir in subdirs:
        file_list = os.listdir(data_path + "/" + subdir)
        files[subdir] = file_list

    for i, type in enumerate(files):
        f = open(data_path+"_label.txt", 'a')
        count = 0
        for file in files[type]:
            file_path = data_path + "/" + type + "/" + file + " " + str(i)
            f.write(file_path)
            f.write('\n')
            count += 1
            if (count == batch):
                print("Finish: " + type)
                break

label("train", 16000)
label("test", 1600)