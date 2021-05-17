from tensorflow.keras.models import load_model
import numpy as np
import imutils
from imutils.contours import sort_contours
import cv2
import argparse
import preprocess

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='Path to input image')
arg = vars(ap.parse_args())
detect_model = load_model("./models/detect.h5")
sum_model = load_model("./models/sum.h5")
image = cv2.imread(arg['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
chars = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape
        if tW > tH:
            thresh = imutils.resize(thresh, width=28)
        else:
            thresh = imutils.resize(thresh, height=28)
        (tH, tW) = thresh.shape
        dX = int(max(0, 28 - tW) / 2.0)
        dY = int(max(0, 28 - tH) / 2.0)
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                    left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))
        padded = cv2.resize(padded, (28, 28))
        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)
        chars.append((padded, (x, y, w, h)))


boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")
preds = detect_model.predict(chars)
labelNames = "0123456789"
labelNames = [l for l in labelNames]
data = []
for (pred, (x, y, w, h)) in zip(preds, boxes):
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]
    data.append(label)
    # print("[INFO] {} - {:.2f}%".format(label, prob * 100))
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.putText(image, label, (x - 10, y - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

max_length = 9
if len(data) % 2 == 0:
    id = len(data) // 2
    data[id] = '+'
else:
    id = (len(data) - 1) // 2
    data[id] = '+'
data = ''.join(x for x in data)
if len(data) != max_length:
    data += ''.join(" " for _ in range(max_length - len(data)))
show_data = data
show_data = show_data.split(" ")
show_data = show_data[0]
print("Your operation is: {}".format(show_data))
data = preprocess.preprocess(data)
data = data.reshape(1, max_length, 12)
result = sum_model.predict(data)
predict = [preprocess.result_preprocess(x) for x in result]
predict = predict[0]
print("Result: {} = {}".format(show_data, predict))
