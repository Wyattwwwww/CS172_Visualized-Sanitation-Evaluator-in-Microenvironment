import numpy as np
import cv2 as cv

def meanShift(points):
    MAXLOOP = 1000
    THRESHOLD = 500

    center_map = {}
    classes = []
    classes_num = 0

    for point in points:
        center = point
        loop_count = 1
        while True:
            if loop_count > MAXLOOP:
                break

            point_count = 1
            new_center = center.copy()

            for p in points:
                if (p[0]-point[0])**2 + (p[1]-point[1])**2 < THRESHOLD*2:
                    new_center[0] += p[0]
                    new_center[1] += p[1]
                    point_count += 1

            new_center[0] = new_center[0] // point_count
            new_center[1] = new_center[1] // point_count
            loop_count += 1
            if new_center == center:
                break

            center = new_center.copy()

        for key, value in center_map.items():
            if (key[0]-center[0])**2 + (key[1]-center[1])**2 < THRESHOLD:
                center_map[(center[0], center[1])] = value
                break

        if (center[0], center[1]) not in center_map.keys():
            classes_num += 1
            center_map[(center[0], center[1])] = classes_num
            print(classes_num)
            classes.append([])

        classes[center_map[(center[0], center[1])]-1].append(point)

    return classes

