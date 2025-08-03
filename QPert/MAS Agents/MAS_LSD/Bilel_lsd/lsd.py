from MAS_LSD.Bilel_lsd.k_means import *
import random
from MAS_LSD.Bilel_lsd.distance import *
import time
np.random.seed(40)
np.seterr(divide='ignore', invalid='ignore')

r = 0.3  # r is the ra-neighboorhood of  point in a shape (set to 0.3, we can also 0.1, 0.2, 0.5 etc)
n = 0  # number of sampled points. Decrease the number to speed up algorithm
k = 40
binss = 32


# calcul the distance between a point and other point in its r-neighborhood
def distances_in_region_of_pt(point, shape):
    distances = []
    # other points in the database
    for other in shape:
        dist = np.linalg.norm(point - other)
        # if the point other is in the region of rayon r of point
        if dist <= r:
            distances.append(dist)
    return distances


# Calculate histogram in r-neighborhood of a point
def histograms_for_a_point(distances_in_r):
    hist, bins = np.histogram(distances_in_r, bins=binss)
    return bins, hist


# Clustering of histograms to get less than 2048  -> 135
def clustering_hists(histograms):
    k_histgms = []
    # print("\nClustering normalized data with k=" + str(k))
    clustering = cluster(np.array(histograms), k)

    # print("\nk-clusters of histograms: ")
    k_histgms = display(np.array(histograms), clustering, k)
    # clus = 1
    # for h in k_histgms:
    #   print("Cluster" , clus, ": ")
    #  print("\n",  h)
    #   clus = clus + 1

    # normalize histogramms
    for j in range(k):
        k_histgms[j] = normalize_vector(k_histgms[j])

    return k_histgms

def point_shape_distance(point, points):
    point_ = np.full((2048, 3), point)
    point_ = np.subtract(point_, points)
    point_ = np.power(point_, 2)
    point_ = np.sum(point_, axis=1)
    point_ = np.sqrt(point_)
    return point_[(0.3 > point_)]

# The LSD descriptor for a given 3D object "shape"
def LSD_descriptor(shape):
    hists = []
    # t0 = time.time()
    for i in range(2048):
        distance = point_shape_distance(shape[i], shape)
        bins, hist = np.histogram(distance, bins=31)
        hists.append(hist)
    hists = np.asarray(hists)
    # print("Creat hitograms   : ", time.time() - t0)
    # t0 = time.time()
    k_hist = clustering_hists(hists)
    # print("Creat clustring   : ", time.time() - t0)
    return np.asarray(k_hist)


# from Mas_PoinNet.MAS.Helper import Load_H5_Dataset

# Read 3D objects. coordinates are points in each point in the database, lables are their corresponding names
# coordinates, labels = Load_H5_Dataset()
#
# # a 3D shape in the database with index 9804
# sample = coordinates[1000]
# shape2 = coordinates[2]

# n is the number of sampled points. To speed up calculation, we reduce the original size, in our case n= 2048
# n= len(shape)
# n = 2048
# # k represents 1% of the total number of points
# k = 2
# # Calculate LSD descriptors for the given shapes. It returns K histograms
# import time
#
# t1 = time.time()
# x= LSD_descriptor(sample)
# x = np.asarray(x)
# print(x.shape)
# print(x)
# for l in x :
#     print(l)
# print("processing  : ", time.time() - t1)
# hitgms2 = LSD_descriptor(shape2, 200)
# print(hitgms)
# print(hitgms.shape)
# # # distance between a shape and its self, should be 0
# distance = optimal_hungarian(hitgms,hitgms2)
# print(distance)
# dist = np.linalg.norm(hitgms-hitgms)
# print(type(dist))
