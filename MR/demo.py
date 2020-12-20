import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter

# ===================================== prepare data
image = cv2.imread("./data_train/digits.png", 0)
data_train = []
for i in range(0, 1000, 20):
    for j in range(0, 2000, 20):
        data_train.append(image[i:i + 20, j:j + 20].reshape(-1))

data_train = np.asarray(data_train)
print(data_train.shape)

# ====================================== model
model = KMeans(n_clusters=30)
model.fit(data_train)
label = model.predict(data_train)
# map_label = {}
# for i in range(0, 600, 60):
#     map_label[Counter(label[i:i + 30]).most_common(1)[0][0]] = i // 60

center = model.cluster_centers_
# for i in range(30):
#     a = center[i].reshape(20,20)
#     a = np.asarray(a, dtype=np.int8)
#     cv2.imshow(str(i), a)
#     cv2.waitKey()


# ======================================


# ===============================================test
test = cv2.imread("./data_test/digit_test3.png")
gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    digit = gray[y - 10:y + h + 20, x - 10:x + w + 20]
    cv2.imshow("origin", digit)
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    digit = 255 - digit
    flat = digit.reshape(-1)
    cv2.imshow("digit", digit)
    predict = model.predict([flat])[0]
    pre = center[predict].reshape(20,20)
    pre = np.asarray(pre,dtype=np.int8)
    cv2.imshow("res", pre)
    print(predict)
    cv2.waitKey()




