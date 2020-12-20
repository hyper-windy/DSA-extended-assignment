import numpy as np
import cv2
from sklearn.cluster import KMeans

# ==========================================
n_clusters = 4
# ==========================================
image = cv2.imread("./data_train/trong_vu.jpg")
data = image.reshape((-1,3))
# ==========================================
model = KMeans(n_clusters=n_clusters)
model.fit(data)
label = model.predict(data)
# ===========================================
dst = np.zeros_like(data)
for k in range(n_clusters):
    dst[label==k] = model.cluster_centers_[k]

dst = dst.reshape(image.shape)
cv2.imshow("Output", dst)
cv2.waitKey()