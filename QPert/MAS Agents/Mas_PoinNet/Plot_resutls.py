from Mas_PoinNet.MAS.Helper import *
from glob import glob


def ModelNet_40_sampled():
    # load all points and labels
    path = path_to_modelnet
    path = os.path.join(path, "modelnet40\\data")
    filenames = ['5.h5', '6.h5', '1.h5', '2.h5', '3.h5', '4.h5', '0.h5']
    points = None
    labels = None
    for d in filenames:
        cur_points, cur_labels = load_h5(os.path.join(path, d))
        cur_points = cur_points.reshape(1, -1, 3)
        cur_labels = cur_labels.reshape(1, -1)
        print(cur_points.reshape(-1, 2048, 3).shape)
        # print(d , np.asarray(cur_points).reshape(-1, 2048, 3).shape)
        if labels is None or points is None:
            labels = cur_labels
            points = cur_points
        else:
            labels = np.hstack((labels, cur_labels))
            points = np.hstack((points, cur_points))

    points_r = points.reshape(-1, 2048, 3)
    labels_r = labels.reshape(-1, 1)
    return points_r, labels_r


def load_names():
    path = path_to_modelnet
    path = os.path.join(path, "modelnet40\\names")
    filenames = ['5.json', '6.json', '1.json', '2.json', '3.json', '4.json', '0.json']
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


def Load_npy_files(path):
    files = os.listdir(path)
    result = None
    for file in files:
        data = np.load(os.path.join(path, file), allow_pickle=True)
        if result is None:
            result = data
        else:
            result = np.hstack((result, data))
    return result


def ModelNet_40_Off_links():
    link = "C:\\Users\\NEW.PC\\Desktop\\datasets\\ModelNet40"
    folders = os.listdir(link)
    result = None
    for folder in folders:
        path = os.path.join(link, folder)
        train_paths = glob(path + "/train/*.off")
        test_paths = glob(path + "/test/*.off")
        if result is None:
            result = np.asarray(train_paths)
            result = np.hstack((result, test_paths))
        else:
            result = np.hstack((result, train_paths))
            result = np.hstack((result, test_paths))
    return result


def find_object_link(name, links):
    for link in links:
        if name in link:
            return link

fitenss_link = "C:\\Users\\NEW.PC\\Desktop\\res\\fitness"
precision_link = "C:\\Users\\NEW.PC\\Desktop\\res\\precision"

x, y = ModelNet_40_sampled()
names = load_names()

fitness_objects = Load_npy_files(fitenss_link)
precision_objects = Load_npy_files(precision_link)

modelnet_off_links = ModelNet_40_Off_links()

print()

n = 100
obj_link = find_object_link(names[n], modelnet_off_links)
# print(names[n])
# show_pc(x[n])
# show_pc(fitness_objects[n])
# show_pc(precision_objects[n])
#
#
# print(obj_link)
# mesh = trimesh.load(obj_link)
# mesh.show()
#
# # Save images with
#
#
# choosene_indexs = np.random.choice(range(12308), n, replace=False)
# print(choosene_indexs)
