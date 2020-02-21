import cv2
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import os


def create_background(h, w):
    background = np.zeros([h, w, 3], np.uint8)
    background[:, :, 0] = np.zeros([h, w]) + 220
    background[:, :, 1] = np.zeros([h, w]) + 220
    background[:, :, 2] = np.zeros([h, w]) + 220
    return background

def circ_mask(height, width, x, y, r=300):
    mask = np.zeros([height, width], np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    return mask

def make_mask(img, x, y, r):
    ret, thresh = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY_INV)
    mask = circ_mask(img.shape[0], img.shape[1], x, y, r)
    return cv2.bitwise_and(thresh, thresh, mask=mask)

def combine_img_bg(img, background, mask):
    mask_inv = cv2.bitwise_not(mask)
    filtered = cv2.bitwise_and(img, img, mask=mask)
    background = cv2.bitwise_and(background, background, mask=mask_inv)
    return cv2.add(filtered, background)

def make_cluster(img1, img2, x, y, r, e=50):
    x_e = rand.randint(x-e*6, x+e*6)
    y_e = y
    r_e = rand.randint(r-e, r+e)

    img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    mask = make_mask(img, x_e, y_e, r_e)
    return combine_img_bg(img1, img2, mask)

def generate_data(path, files, out_path):
    cluster_num = rand.randint(1, 5)
    type_idx_list = []

    for i in range(cluster_num):
        types = list(files.keys())
        type_idx = rand.randint(0, len(files) - 1)
        type_files = files[types[type_idx]]

        file_idx = rand.randint(0, len(type_files) - 1)
        file = type_files[file_idx]
        file_path = path + "/" + types[type_idx] + "/" + file

        img_raw = cv2.imread(file_path)
        if img_raw is None:
            continue
        if not img_raw.any():
            continue
        type_idx_list.append(type_idx)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        height, width, _ = img_raw.shape

        x = width // 2
        y = (height-300) // cluster_num
        if i == 0:
            background = create_background(height, width)
            global new_data
            new_data = make_cluster(img_raw, background, x, y, 250, 150)
        else:
            new_data = make_cluster(img_raw, new_data, x, (y+y*i), 250, 150)

    data_name = ""
    for idx in type_idx_list:
        if idx + 1 == 7 or idx + 1 == 8:
            idx = 5
        if idx + 1 == 9:
            idx = 6
        if data_name == "":
            data_name = str(idx+1)
        else:
            data_name = data_name + "_" + str(idx+1)
    data_name = data_name + ".jpg"

    new_data = new_data[..., ::-1]
    new_data = cv2.cvtColor(new_data, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(out_path+"/"+data_name, new_data)


data_path = "data"
subdirs = os.listdir(data_path)
files = {}
for subdir in subdirs:
    file_list = os.listdir(data_path + "/" + subdir)
    files[subdir] = file_list

for i in range(100):
    if i % 10 == 0:
        print("Finished: " + str(i))
    generate_data(data_path, files, "mixed")
