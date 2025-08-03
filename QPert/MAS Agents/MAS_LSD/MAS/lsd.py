import numpy as np
import time
from sklearn.cluster import KMeans

np.random.seed(40)


# calculate distance between a point X and all shape objects
def point_shape_distance(point, points):
    point_ = np.full((2048, 3), point)
    point_ = np.subtract(point_, points)
    point_ = np.power(point_, 2)
    point_ = np.sum(point_, axis=1)
    point_ = np.sqrt(point_)
    return point_[(0.3 > point_)]


# calculate LSD discriptor
def Lsd(points):
    n=2048
    # creat a list of histograms
    hists = []
    # choose random points
    indexs = np.random.choice(range(2048), n, replace=False)
    # calculate distances between each choosen points and all shape points
    for i in range(2048):
        distance = point_shape_distance(points[i], points)
        bins, hist = np.histogram(distance, bins=31)
        hists.append(hist)
    hists = np.asarray(hists)
    # get K Histograms using K-means
    kmeans = KMeans(n_clusters=40, random_state=1).fit(hists)
    centers = kmeans.cluster_centers_
    return np.asarray(centers)

#
#Read 3D objects. coordinates are points in each point in the database, lables are their corresponding names
# x, y = Load_H5_Dataset()
# # t0 = time.time()
# # for i in range(10):
# n = 10000
# d= Lsd(x[n])
# print(np.max(d))


# print("Point to shape function took : ", time.time() - t0)

# dis =  np.load("modelnet_lsd_discriptors.npy" , allow_pickle=True)
# test_d = dis[n][1]
# print(test_d[0])