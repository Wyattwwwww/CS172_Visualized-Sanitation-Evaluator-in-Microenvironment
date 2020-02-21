# coding=utf-8
import cv2
import numpy as np
from utils import meanShift

# usage:
# FLASK_APP=app.py python3 -m flask run --host=0.0.0.0

def blob(im):
    size = im.shape
    scale = 0.3
    img = cv2.resize(im, (int(scale * size[1]), int(scale * size[0])), interpolation=cv2.INTER_CUBIC)

    size = img.shape  # 图像大小
    row = size[0]
    column = size[1]
    num_of_suspect = 0  # 可疑点的个数
    suspect = []  # 可疑点集合

    # R G B D 参数
    # R = 120
    # G = 120
    # B = 120
    D = 100

    # 取出rgb通道 滤波 据说这个函数提出来的顺序是bgr
    intensity = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0.1, 0.1)

    # Detect
    # 检测是不是可疑点 假设G和R应该小于一定值 B大于一定值
    MAX_SUSPECT = 9999999
    for x in range(row):
        for y in range(column):
            ixy = int(intensity[x, y])

            diff = 0
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    m = x + dx
                    n = y + dy
                    m = min(max(0, m), row - 1)
                    n = min(max(0, n), column - 1)
                    diff += abs(int(intensity[m, n]) - ixy)

            if diff > D:
                suspect.append([x, y])
                num_of_suspect = num_of_suspect + 1
                if num_of_suspect >= MAX_SUSPECT:
                    break

        if num_of_suspect >= MAX_SUSPECT:
            break
            
    classes = meanShift(suspect)
    num_of_classes = len(classes)

    # 绘制标记框
    count = 0
    rects = []
    for i in range(num_of_classes):
        if len(classes[i]) > 50:
            cnt = np.array(classes[i])
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x,y,w,h))
            count += 1

    text = 'Suspect: {}'.format(count)
    org = (column - 110, row - 20)
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 0.5
    fontcolor = (0, 0, 255)  # BGR
    thickness = 1
    lineType = 4
    cv2.putText(img, text, org, fontFace, fontScale, fontcolor, thickness, lineType)
    return rects, img


# Input:  numpy in BGR form
# Output: numpy in BGR form
def fluorescence(img):
    size = img.shape
    scale = 0.3
    img = cv2.resize(img, (int(scale*size[1]), int(scale*size[0])), interpolation=cv2.INTER_CUBIC)

    size = img.shape  # 图像大小
    row = size[0]
    column = size[1]

    num_of_suspect = 0  # 可疑点的个数
    suspect = []  # 可疑点集合

    # R G B 参数
    R = 150
    G = 160
    B = 120
    D = 40

    # 取出rgb通道 滤波 据说这个函数提出来的顺序是bgr
    red = cv2.GaussianBlur(img[:, :, 2], (3, 3), 0.2, 0.2)
    green = cv2.GaussianBlur(img[:, :, 1], (3, 3), 0.2, 0.2)
    blue = cv2.GaussianBlur(img[:, :, 0], (3, 3), 0.2, 0.2)
    intensity = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (3, 3), 0.1, 0.1)

    # Detect
    # 检测是不是可疑点 假设G和R应该小于一定值 B大于一定值
    MAX_SUSPECT = 99999
    for x in range(row):
        for y in range(column):
            b = int(blue[x, y])
            r = int(red[x, y])
            g = int(green[x, y])
            i = int(intensity[x, y])

            diff = 0
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    m = x + dx
                    n = y + dy
                    m = min(max(0, m), row-1)
                    n = min(max(0, n), column-1)
                    diff += abs(int(intensity[m, n]) - i)

            if diff > D and b >= (r + g)*0.55 and b > B and r < R and g < G:
                suspect.append([x, y])
                num_of_suspect = num_of_suspect + 1
                if num_of_suspect >= MAX_SUSPECT:
                    break

        if num_of_suspect >= MAX_SUSPECT:
            break

    # 靠近的点聚类
    # Cluster
    # inlier参数 Eu距离的平方（像素^2）
    inlier = 1000

    num_of_classes = 0
    classes = []
    if num_of_suspect != 0:
        for i in range(num_of_suspect):
            if i == 0:
                x = suspect[i][0]
                y = suspect[i][1]
                classes.append([x, y, 1])
                num_of_classes = 1
            else:
                for j in range(num_of_classes):
                    if (suspect[i][0] - classes[j][0]) ** 2 + (suspect[i][1] - classes[j][1]) ** 2 <= inlier:
                        classes[j][2] += 1
                        break
                else:
                    x = suspect[i][0]
                    y = suspect[i][1]
                    classes.append([x, y, 1])
                    num_of_classes += 1

    # 可疑点矩阵 [x坐标 y坐标 标记圈半径]
    # position = np.zeros([num_of_classes, 3], dtype=int)
    for i in range(num_of_classes):
        classes[i][2] = int(classes[i][2] * 0.1 + 10)

    # 绘制标记圈
    count = 0
    for i in range(num_of_classes):
        if classes[i][2] < 50:
            cv2.circle(img, (classes[i][1], classes[i][0]), classes[i][2], (0, 0, 255), 1)
            count = count + 1

    # 报告可疑点数量
    text = 'Suspect: {}'.format(count)
    org = (column - 110, row - 20)
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 0.5
    fontcolor = (0, 0, 255)  # BGR
    thickness = 1
    lineType = 4
    cv2.putText(img, text, org, fontFace, fontScale, fontcolor, thickness, lineType)
    return img