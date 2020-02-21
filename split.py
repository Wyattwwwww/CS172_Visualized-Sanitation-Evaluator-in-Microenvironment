import cv2
import os

data_path = "data"
subdirs = os.listdir(data_path)
files = {}
for subdir in subdirs:
    file_list = os.listdir(data_path + "/" + subdir)
    files[subdir] = file_list

types = list(files.keys())
for i in range(len(files)):
    if i+1 == 7 or i+1 == 8:
        continue
    count1 = 0
    count2 = 0
    type = types[i]
    for j, file in enumerate(files[type]):
        file_path = data_path + "/" + type + "/" + file
        img = cv2.imread(file_path)
        if img is None:
            continue
        if not img.any():
            continue
        height, width, _ = img.shape
        for k in range(4,17):
            h = height // k
            w = width // k
            for m in range(k):
                for n in range(k):
                    img_out = img[h*m:h*(m+1), w*n:w*(n+1), :]
                    img_out = cv2.resize(img_out,(width//8, height//8),interpolation=cv2.INTER_NEAREST)
                    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
                    if j<4:
                        if count1 < 320:   # max size of test data
                            dir = "small_dataset/test/" + str(i+1) + "." + type
                            if i+1 == 9:
                                dir = "small_dataset/test/" + "7" + "." + type
                            count1 += 1
                        else:
                            break
                    else:
                        if count2 < 3200:  # max size of train data
                            dir = "small_dataset/train/" + str(i+1) + "." + type
                            if i+1 == 9:
                                dir = "small_dataset/train/" + "7" + "." + type
                            count2 += 1
                        else:
                            break
                    exist = os.path.exists(dir)
                    if not exist:
                        os.makedirs(dir)
                    out_path = dir + "/" + str(j) + "-" + str(k) + "_" + str(m) + "_" + str(n) + ".jpg"
                    cv2.imwrite(out_path, img_out)

