from microbio import *
import matlab.engine
import os
import random
import re
import pandas as pd
import torchvision.transforms as transforms
import torch as t
import myModels
from PIL import Image

def classifier(img_path):
    ''' BoW model. '''
    # return eng.classifier(img_path)

    ''' SPPNet model. '''
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    model = myModels.SPPNet(7)
    model.load_state_dict(t.load('model_parameters_SPPNet.pt'))
    model = model.to(device)
    model.eval()

    img = Image.open(img_path)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(150),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    img = img.convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with t.no_grad():
        py = model(img)
    _, predicted = t.max(py, 1)
    return predicted.item() + 1

# eng = matlab.engine.start_matlab()


path = "mixed"
out_path = "SPPNet"

files = os.listdir(path)
test_set = random.sample(files, 10)
accuracy = []
detected = [[],[],[],[],[],[],[]]

for file in test_set:
    img = cv2.imread(path + "/" + file)
    rects, img = blob(img)                        # Detection.

    idx = re.split(r'[_.\s]\s*', file)
    idx = [int(i) for i in idx if i.isdigit()]
    result = []

    for i, rect in enumerate(rects):
        x, y, w, h = rect
        cv2.imwrite(str(i)+".jpg", img[x:x+w, y:y+h, :])

        label = classifier(str(i) + ".jpg")       # Classify.
        result.append(label)

        text = str(label)
        org = (y,x+10)
        fontFace = cv2.FONT_HERSHEY_TRIPLEX
        fontScale = 0.4
        fontcolor = (0, 0, 255)
        thickness = 1
        lineType = 4
        cv2.putText(img, text, org, fontFace, fontScale, fontcolor, thickness, lineType)

        os.remove(str(i)+".jpg")

    for i, rect in enumerate(rects):
        x, y, w, h = rect
        cv2.rectangle(img, (y, x), (y + h, x + w), (0, 0, 255), 1)

    exist = os.path.exists(out_path)
    if not exist:
        os.makedirs(out_path)
    cv2.imwrite(out_path + "/" + file, img)

    # Save the results.
    count = 0
    for i in idx:
        if i >= 7:
            if 7 in result:
                count += 1
        else:
            if i in result:
                count += 1
    accuracy.append(count/len(idx))

    for i in range(7):
        if i+1 not in idx:
            detected[i].append("None")
        else:
            if i+1 in result:
                detected[i].append("1")
            else:
                detected[i].append("0")

dataframe = pd.DataFrame({'Filename': test_set,
                          'Accuracy': accuracy,
                          'Detected Type 1': detected[0],
                          'Detected Type 2': detected[1],
                          'Detected Type 3': detected[2],
                          'Detected Type 4': detected[3],
                          'Detected Type 5': detected[4],
                          'Detected Type 6': detected[5],
                          'Detected Type 7': detected[6],
                          })
dataframe.to_csv(out_path+"/result.csv", mode='a', index=False, sep=',')