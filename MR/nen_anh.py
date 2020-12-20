import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==========================================

# ==========================================
image = cv2.imread("./data_train/trong_vu.jpg")
data = image.reshape((-1,3))
N_Clusters = [5,10,15,20]
res = []
for n_clusters in N_Clusters:
    model = KMeans(n_clusters=n_clusters)
    model.fit(data)
    label = model.predict(data)
    dst = np.zeros_like(data)
    for k in range(n_clusters):
        dst[label==k] = model.cluster_centers_[k]

    dst = dst.reshape(image.shape)
    res.append(dst)

for i in range(2):
    for j in range(2):
        plt.subplot(2,2, 2*i+j+1)
        dst = cv2.cvtColor(res[2*i+j], cv2.COLOR_BGR2RGB)
        plt.imshow(dst)
        plt.title("n_clusters = "+str(N_Clusters[2*i+j]))
        plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey()