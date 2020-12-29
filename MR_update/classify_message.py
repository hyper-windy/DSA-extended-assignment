from sklearn.cluster import KMeans
import  numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import cv2

data_file ="./train-features-50.txt"
n_word = 2500
lb = np.array([[0]*25+[1]*25]).reshape(-1)


data = np.zeros((50, 2500),dtype=np.float32)
with open(data_file) as f:
    for line in f.readlines():
        msg, id, num = map(int, line.split())
        data[msg-1][id-1] = num

numberOfClusters = 2
stopCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
ret, labels, clusters = cv2.kmeans(data, numberOfClusters, None, stopCriteria, 10, cv2.KMEANS_PP_CENTERS)
print(labels.reshape(-1))

#########
clf = MultinomialNB()
clf.fit(data, lb)
y_pred = clf.predict(data)
print(y_pred)