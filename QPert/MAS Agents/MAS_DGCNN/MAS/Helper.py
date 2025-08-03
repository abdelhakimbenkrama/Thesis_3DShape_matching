import numpy as np
import os
import h5py

path_to_modelnet = "S:\\Phd Work\\MAS work\\data\\"

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
    # labels_r = np_utils.to_categorical(labels_r, 40)
    return points_r, labels_r

def load_discriptors(path):
    data = np.load(os.path.join(path ,"modelnet40_dgcnn_features_2.npy") ,allow_pickle=True)
    return data
    vectors = data[: , 0]
    norm_disc = []
    # normalize the dataset
    for e in vectors:
        norm = np.linalg.norm(e)
        norm_e = e / norm
        norm_disc.append(norm_e)
    disc = np.array(norm_disc)
    labels = data[: , 1]
    return np.c_[disc, labels]

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

def norm_matrix(mat):
    norm_mat = []
    for e in mat:
        norm = np.linalg.norm(e)
        norm_e = e / norm
        norm_mat.append(norm_e)
    return np.array(norm_mat)

path_to_discriptors = "S:\\Phd Work\\MAS work\\data\\DGCNN_descriptors\\"

objects_Discriptors_data = load_discriptors(path_to_discriptors)
x,y = Load_H5_Dataset()
print(type(objects_Discriptors_data[0][0][0]))
print(objects_Discriptors_data[0][0][0])