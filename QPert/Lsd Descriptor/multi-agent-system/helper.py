import numpy as np
from sklearn.cluster import KMeans

def Two_Vectors_CrosseOver_Mutation(a, b):
    out = np.zeros((512, 3))
    a_son = np.zeros((512, 3))
    b_son = np.zeros((512, 3))

    for ind in range(512):
        a_son[ind] = a[ind]
        b_son[ind] = b[ind]
    # 164
    lenght = int(512 * 8 / 100) + 1
    # Mutation with Q
    indexs = np.random.choice(range(512), lenght, replace=False)

    for m in indexs:
        a_son[m] = b[m]
        b_son[m] = a[m]

    # CrosseOver with Q
    crossover_points = np.random.choice(range(512), 2, replace=False)
    crossover_points = np.sort(crossover_points)
    # Permutation
    out[:crossover_points[0]] = a_son[:crossover_points[0]]
    out[crossover_points[1]:] = a_son[crossover_points[1]:]

    a_son[:crossover_points[0]] = b_son[:crossover_points[0]]
    a_son[crossover_points[1]:] = b_son[crossover_points[1]:]

    b_son[:crossover_points[0]] = out[:crossover_points[0]]
    b_son[crossover_points[1]:] = out[crossover_points[1]:]
    return a_son, b_son


def norm_matrix(mat):
    norm_mat = []
    for e in mat:
        norm = np.linalg.norm(e)
        norm_e = e / norm
        norm_mat.append(norm_e)
    return np.array(norm_mat)

def point_shape_distance(point, points):
    point_ = np.full((512, 3), point)
    point_ = np.subtract(point_, points)
    point_ = np.power(point_, 2)
    point_ = np.sum(point_, axis=1)
    point_ = np.sqrt(point_)
    return point_[(0.3 > point_)]

def LSD(points):
    n = 512
    # creat a list of histograms
    hists = []
    # normalize the points
    norm = np.linalg.norm(points)
    points = points / norm
    # calculate distances between each choosen points and all shape points
    for i in range(n):
        distance = point_shape_distance(points[i], points)
        bins, hist = np.histogram(distance, bins=31)
        hists.append(hist)
    hists = np.asarray(hists)
    # get K Histograms using K-means
    kmeans = KMeans(n_clusters=40, random_state=0).fit(hists)
    centers = kmeans.cluster_centers_
    return np.asarray(centers)
