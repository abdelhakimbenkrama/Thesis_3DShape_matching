import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import json

np.random.seed(40)

# PointNet Discriptors Links
PointNet_Discriptors_Path = "S:\\Phd Work\\MAS work\\data\\PointNet_Descriptors.npy"
PointNet_Labels_Path = "S:\\Phd Work\\MAS work\\data\\PointNet_Descriptors_labels.npy"
# Lsd
Lsd_Discriptors_Path = "S:\\Phd Work\\MAS work\\data\\Lsd_Descriptors"
# model net contained folder
path_to_modelnet = "S:\\Phd Work\\MAS work\\data\\"


def Two_Vectors_CrosseOver_Mutation(a, b):
    out = np.zeros((2048, 3))
    a_son = np.zeros((2048, 3))
    b_son = np.zeros((2048, 3))

    for ind in range(2048):
        a_son[ind] = a[ind]
        b_son[ind] = b[ind]
    # 164
    lenght = int(2048 * 8 / 100) + 1
    # Mutation with Q
    indexs = np.random.choice(range(2048), lenght, replace=False)

    for m in indexs:
        a_son[m] = b[m]
        b_son[m] = a[m]

    # CrosseOver with Q
    crossover_points = np.random.choice(range(2048), 2, replace=False)
    crossover_points = np.sort(crossover_points)
    # Permutation
    out[:crossover_points[0]] = a_son[:crossover_points[0]]
    out[crossover_points[1]:] = a_son[crossover_points[1]:]

    a_son[:crossover_points[0]] = b_son[:crossover_points[0]]
    a_son[crossover_points[1]:] = b_son[crossover_points[1]:]

    b_son[:crossover_points[0]] = out[:crossover_points[0]]
    b_son[crossover_points[1]:] = out[crossover_points[1]:]
    return a_son, b_son


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def Load_H5_Dataset():
    # load all points and labels
    path = path_to_modelnet
    path = os.path.join(path, "modelnet40\\data")
    filenames = [d for d in os.listdir(path)]
    points = None
    labels = None
    for d in filenames:
        cur_points, cur_labels = load_h5(os.path.join(path, d))

        cur_points = cur_points.reshape(1, -1, 3)
        cur_labels = cur_labels.reshape(1, -1)
        #print(d , np.asarray(cur_points).reshape(-1, 2048, 3).shape)
        if labels is None or points is None:
            labels = cur_labels
            points = cur_points
        else:
            labels = np.hstack((labels, cur_labels))
            points = np.hstack((points, cur_points))
    points_r = points.reshape(-1, 2048, 3)
    labels_r = labels.reshape(-1, 1)
    return points_r, labels_r


def Load_PointNet_Discriptors():
    Data_Path = PointNet_Discriptors_Path
    Labels_Path = PointNet_Labels_Path
    disc = np.load(Data_Path)
    norm_disc = []
    # normalize the dataset
    for e in disc:
        norm = np.linalg.norm(e)
        norm_e = e / norm
        norm_disc.append(norm_e)
    disc = np.array(norm_disc)
    Labels = np.load(Labels_Path)
    # join data
    return np.c_[disc, Labels]

def Load_LSD_Discriptors():
    Data_Path = Lsd_Discriptors_Path
    Labels_Path = PointNet_Labels_Path
    disc = np.load(Data_Path)
    norm_disc = []
    # normalize the dataset
    for e in disc:
        norm = np.linalg.norm(e)
        norm_e = e / norm
        norm_disc.append(norm_e)
    disc = np.array(norm_disc)
    Labels = np.load(Labels_Path)
    # join data
    return np.c_[disc, Labels]

def Load_Lsd_Discriptors():
    files = os.listdir(Lsd_Discriptors_Path)
    data = None
    for file in files:
        if data is None:
            data = np.load(os.path.join(Lsd_Discriptors_Path, file), allow_pickle=True)
        else:
            z = np.load(os.path.join(Lsd_Discriptors_Path, file), allow_pickle=True)
            data = np.append(data, z, axis=0)
    return data


# d = Load_Lsd_Discriptors()
# print(d.shape)
# x = d[0]
# print(x[0].shape)
# print(x[1].shape)


def norm_matrix(mat):
    norm_mat = []
    for e in mat:
        norm = np.linalg.norm(e)
        norm_e = e / norm
        norm_mat.append(norm_e)
    return np.array(norm_mat)


def plot_point_clouds_obj(arr):
    ax = plt.axes(projection='3d')
    ax.scatter(arr[:, 0] * 100, arr[:, 1] * 100, arr[:, 2] * 100, s=0.01)
    plt.show()


def load_names():
    path = path_to_modelnet
    path = os.path.join(path, "modelnet40\\names")
    filenames = [d for d in os.listdir(path)]
    names = None
    for d in filenames:
        file = open(os.path.join(path, d))
        data = json.load(file)
        data = np.asarray(data)
        if names is None:
            names = data
        else:
            names = np.hstack((names, data))
    only_names = []
    for name in names:
        n = os.path.basename(os.path.normpath(name))
        only_names.append(n[:-4])
    return np.asarray(only_names)




#
# x, y = Load_H5_Dataset()
#
# X = x[5]
def show_pc(X):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    ax.set_axis_off()
    plt.show()
