import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = []
label = []
# Load original image
originalImage = cv.imread("./data_train/cat.jpg")
originalImage = cv.cvtColor(originalImage, cv.COLOR_BGR2RGB)
originalImage = cv.resize(originalImage, None, fx=0.5, fy=0.5)

img.append(originalImage)
label.append("Original")
# Kmeans
reshapedImage = np.float32(originalImage.reshape(-1, 3))


numberOfClusters = 2
stopCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)
ret, labels, clusters = cv.kmeans(reshapedImage, numberOfClusters, None, stopCriteria, 10, cv.KMEANS_RANDOM_CENTERS)
clusters = np.uint8(clusters)

intermediateImage = clusters[labels.flatten()]
clusteredImage = intermediateImage.reshape((originalImage.shape))

img.append(clusteredImage)
label.append("Clustered")
cv.imshow("sdfg", clusteredImage)
cv.waitKey()
# =================================================
removedCluster = 1

cannyImage = np.copy(originalImage).reshape((-1, 3))
cannyImage[labels.flatten() == removedCluster] = [0, 0, 0]

cannyImage = cv.Canny(cannyImage, 100, 200).reshape(originalImage.shape)

initialContoursImage = np.copy(cannyImage)
imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(imgray, 50, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(initialContoursImage, contours, -1, (0, 0, 255), cv.CHAIN_APPROX_SIMPLE)

img.append(initialContoursImage)
label.append("initialContoursImage")


cv.imshow("sdfg", initialContoursImage)
cv.waitKey()
cnt = contours[0]
largest_area = 0
index = 0
for contour in contours:
    if index > 0:
        area = cv.contourArea(contour)
        if (area > largest_area):
            largest_area = area
            cnt = contours[index]
    index = index + 1

biggestContourImage = np.copy(originalImage)
cv.drawContours(biggestContourImage, [cnt], -1, (0, 0, 255), 6)
cv.imshow("sdfg", biggestContourImage)
cv.waitKey()
img.append(biggestContourImage)
label.append("result")

b = cv.resize(initialContoursImage, None, fx =0.5, fy =0.5)
c = cv.resize(biggestContourImage, None, fx =0.5, fy =0.5)
a = np.hstack((b, c))
cv.imshow("a",a)
cv.waitKey()
# ===============================================================
for i in range(4):
    plt.subplot(2,2, i+1)
    plt.imshow(img[i])
    plt.title(label[i])
    plt.xticks([]), plt.yticks([])
plt.show()
