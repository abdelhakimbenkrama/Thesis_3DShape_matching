import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from multiprocessing import Pool
from Mas_PoinNet.MAS.classes import RetrievalAgent, CoordinateAgent
from Mas_PoinNet.MAS.Helper import Load_H5_Dataset, Load_PointNet_Discriptors
import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings('ignore')

# LOAD 3D Objects DataSet + labels
x, y = Load_H5_Dataset()
# Load Discriptors  + labels
objects_Discriptors_data = Load_PointNet_Discriptors()

# path to the pretrained weights of pointnet model
Path_To_Weights = "S:\\Phd Work\\MAS work\\data\\pointNet_weights.h5"

Element_per_class = [725, 156, 615, 193, 672, 435, 84, 297, 989, 187,
                     99, 157, 286, 129, 286, 169, 271, 255, 165, 144,
                     169, 384, 565, 286, 108, 331, 339, 124, 215, 148,
                     780, 144, 110, 492, 183, 444, 367, 575, 107, 123]

Element_per_class = np.asarray(Element_per_class)
# PARAMETERS
N = 10
GENERATIONS = 20

# results
results = []


# choose random objects in dataset
# choosene_indexs = np.random.choice(range(12308), TEST_ELEMENTS_NUMBER, replace=False)
# print(choosene_indexs)


def processing(ele_index):
    # track data
    # precesion_per_one_object = np.zeros((N, GENERATIONS))
    # fitness_per_one_object = np.zeros((N, GENERATIONS))

    # create cordination Agent
    Query_cmp_limits = int(np.squeeze(Element_per_class[y[ele_index]]))
    C_agent = CoordinateAgent(x[ele_index], y[ele_index])
    C_agent.get_Point_Net_Model(Path_To_Weights)
    C_agent.creat_clones(N)
    C_agent.discriptor_maker_object()
    C_agent.discriptor_maker_clones()

    # Original Object Fitness
    R_agent_for_original_obj = RetrievalAgent(C_agent.Object_discriptor, C_agent.classe, Query_cmp_limits)
    R_agent_for_original_obj.oredered_distance_list(objects_Discriptors_data)
    R_agent_for_original_obj.precision_func()
    R_agent_for_original_obj.fitness_func()
    C_agent.object_precision_fitness_data = [R_agent_for_original_obj.query, R_agent_for_original_obj.precision,
                                             R_agent_for_original_obj.fitness]
    # THIS PART SHOULD EXECUTED AS EVENTS
    for i in range(GENERATIONS):
        for j, dis in enumerate(C_agent.Discriptors):
            R_agent = RetrievalAgent(dis, C_agent.classe, Query_cmp_limits)
            R_agent.oredered_distance_list(objects_Discriptors_data)
            R_agent.precision_func()
            R_agent.fitness_func()
            C_agent.save_best_precision_clone([C_agent.Clones[j], R_agent.precision])
            C_agent.save_best_fitness_clone([C_agent.Clones[j], R_agent.fitness])

            # chekpoint of precision and fitness of the each clone
            # precesion_per_one_object[j, i] = R_agent.precision
            # fitness_per_one_object[j, i] = R_agent.fitness
        C_agent.generation += 1
        C_agent.CrosseOver_Mutation()
        C_agent.discriptor_maker_clones()

    return ([int(ele_index),
             float("{:.6f}".format(C_agent.object_precision_fitness_data[1])),
             float("{:.6f}".format(C_agent.best_precision_clone_data[1])),
             int(C_agent.best_precision_clone_data[2]),
             float("{:.6f}".format(C_agent.object_precision_fitness_data[2])),
             float("{:.6f}".format(C_agent.best_fitness_clone_data[1])),
             int(C_agent.best_fitness_clone_data[2]),
             np.asarray(C_agent.best_precision_clone_data[0]),
             np.asarray(C_agent.best_fitness_clone_data[0])])


if __name__ == '__main__':
    t1 = time.time()
    p = Pool()
    results = p.map(processing, range(9840, 9844))
    p.close()
    p.join()
    print("processing  : ", time.time() - t1)

    # Save results
    res = np.array(results, dtype="O")
    pre_data = res[:, 7]
    fit_data = res[:, 8]
    info = res[:, :7]
    # print(res.shape)
    print(pre_data.shape)
    print(fit_data.shape)
    print(info.shape)
    # convert your array into a dataframe

    info_df = pd.DataFrame(info,
                           columns=['Object index', 'Object CNN precision', 'Best Clone precision', 'Clone generation',
                                    'Object CNN fitness 1 ', 'Best Clone fitness 1 ', 'Clone generation'])
    ## save to xlsx file
    info_filepath = 'info.xlsx'
    pre_filepath = 'precision_objects.npy'
    fit_filepath = 'fitness_objects.npy'

    info_df.to_excel(info_filepath, index=False, header=True)
    np.save(pre_filepath, pre_data)
    np.save(fit_filepath, fit_data)

"""
[0.         0.00965218 0.01930436 0.02895654 0.03860872 0.0482609
 0.05791308 0.06756526 0.07721744 0.08686962 0.0965218  0.10617398
 0.11582616 0.12547834 0.13513052 0.1447827  0.15443489 0.16408706
 0.17373924 0.18339142 0.1930436  0.20269579 0.21234797 0.22200014
 0.23165232 0.2413045  0.25095668 0.26060885 0.27026105 0.27991322
 0.2895654  0.29921758]
"""