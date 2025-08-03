import os
import pandas as pd
import numpy as np
import h5py

path_to_modelnet = "C:\\Users\\NEW.PC\\Desktop\\data\\"

l= "C:\\Users\\NEW.PC\\Desktop\\datasets\\ModelNet40"
classes = os.listdir(l)

def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def ModelNet_40_sampled():
    path_to_modelnet = "C:\\Users\\NEW.PC\\Desktop\\data\\"
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


def load_data(link):
    files = os.listdir(link)
    data = None
    for file in files:
        one_file_data = pd.read_excel(os.path.join(link, file)).values
        one_file_data = np.asarray(one_file_data)
        if data is None:
            data = one_file_data
        else:
            data = np.concatenate((data, one_file_data), axis=0)
    return data


def avrg(values, labels, classId):
    total = 0
    count = 0
    for i, v in enumerate(values):
        if int(labels[i]) == int(classId):
            total += v
            count += 1
    return count ,total , total/count


# read results files
path = "C:\\Users\\NEW.PC\\Desktop\\res\\data"

data = load_data(path)
lables = ModelNet_40_sampled()[1]

# calculate avarege precision and fitness for each class
result = None

# for i in range(40):
#    print(avrg(data[:, 1], lables, i))
# for i in range(40):
#     pointnetPrecision = avrg(data[:, 1], lables, i)
#     pointnetFitness = avrg(data[:, 4], lables, i)
#     masPrecision = avrg(data[:, 2], lables, i)
#     masfitness = avrg(data[:, 5], lables, i)
#     if result is None:
#         result = [[classes[i], pointnetPrecision, masPrecision, pointnetFitness, masfitness]]
#     else:
#         result.append([ classes[i], pointnetPrecision, masPrecision, pointnetFitness, masfitness])
# result= np.asarray(result)
# print(result.shape)
# print(result)
#
# # save a xlsx file [ class id , pointnet precision , pointnet fitness , Mas_PoinNet precision , Mas_PoinNet fitness ]
# info_df = pd.DataFrame(result,
#                        columns=['class ', 'pointnet precision', 'Mas_PoinNet precision', 'pointnet fitness',
#                                 'Mas_PoinNet fitness'])
# ## save to xlsx file
# info_filepath = 'final_results.xlsx'
#
# info_df.to_excel(info_filepath, index=False, header=True)

result =[]
for i , d in enumerate(data) :
    if 0 == lables[i]:
        result.append(data[i,1])
print(result)

tot = 0

for x in result:
    tot+=x
print(tot)
print(tot/len(result))