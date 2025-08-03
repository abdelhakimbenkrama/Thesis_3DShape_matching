import os
from multiprocessing import Pool
import multiprocessing
import numpy as np
from MAS_DGCNN.MAS.Helper import Load_H5_Dataset ,load_discriptors
from MAS_DGCNN.MAS.classes import RetrievalAgent ,CoordinateAgent
import time
from tqdm import tqdm
import pandas as pd


# LOAD 3D Objects DataSet + labels
x, y = Load_H5_Dataset()

# Load Discriptors  + labels
path_to_discriptors = "C:\\Users\\NEW.PC\\Desktop\\data\\DGCNN_descriptors\\"
objects_Discriptors_data = load_discriptors(path_to_discriptors)

# path to the pretrained weights of pointnet model
path_to_weights = "C:\\Users\\NEW.PC\\Desktop\\data\\model.2048.t7"


Element_per_class = [725, 156, 615, 193, 672, 435, 84, 297, 989, 187,
                     99, 157, 286, 129, 286, 169, 271, 255, 165, 144,
                     169, 384, 565, 286, 108, 331, 339, 124, 215, 148,
                     780, 144, 110, 492, 183, 444, 367, 575, 107, 123]

Element_per_class = np.asarray(Element_per_class)
# PARAMETERS
N = 10
GENERATIONS = 20

# results
results = [ ]
t1 = time.time()
def processing(ele_index):
    print("process" , ele_index ,"Started")
    Query_cmp_limits = int(np.squeeze(Element_per_class[y[ele_index]]))
    C_agent = CoordinateAgent(x[ele_index] , y[ele_index])
    C_agent.creat_clones(N)
    C_agent.loadModel(path_to_weights)
    C_agent.discriptor_maker_object()
    C_agent.discriptor_maker_clones()

    # Original object fitness
    R_agent_for_original_obj = RetrievalAgent(C_agent.Object_discriptor, C_agent.classe, Query_cmp_limits)
    R_agent_for_original_obj.oredered_distance_list(objects_Discriptors_data)
    R_agent_for_original_obj.precision_func()
    R_agent_for_original_obj.fitness_func()
    C_agent.object_precision_fitness_data = [R_agent_for_original_obj.query, R_agent_for_original_obj.precision,
                                             R_agent_for_original_obj.fitness]
    for i in range(GENERATIONS):
        for j , dis in enumerate(C_agent.Discriptors):
            R_agent = RetrievalAgent(dis, C_agent.classe, Query_cmp_limits)
            R_agent.oredered_distance_list(objects_Discriptors_data)
            R_agent.precision_func()
            R_agent.fitness_func()
            C_agent.save_best_precision_clone([C_agent.Clones[j], R_agent.precision])
            C_agent.save_best_fitness_clone([C_agent.Clones[j], R_agent.fitness])
        C_agent.generation += 1
        C_agent.CrosseOver_Mutation()
        C_agent.discriptor_maker_clones()
        print("GEN" , i)

    return ([int(ele_index),
             float("{:.6f}".format(C_agent.object_precision_fitness_data[1])),
             float("{:.6f}".format(C_agent.best_precision_clone_data[1])),
             int(C_agent.best_precision_clone_data[2]),
             float("{:.6f}".format(C_agent.object_precision_fitness_data[2])),
             float("{:.6f}".format(C_agent.best_fitness_clone_data[1])),
             int(C_agent.best_fitness_clone_data[2]),
             np.asarray(C_agent.best_precision_clone_data[0]),
             np.asarray(C_agent.best_fitness_clone_data[0])])

from numpy import random

indexes =  random.randint(12000 , size=(4))
indexes = list(indexes)


if __name__ == '__main__':
    t1 = time.time()
    p = Pool(multiprocessing.cpu_count())
    results = p.map(processing, [0])
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
    # info_df = pd.DataFrame(info,
    #                        columns=['Object index', 'Object CNN precision', 'Best Clone precision', 'Clone generation',
    #                                 'Object CNN fitness 1 ', 'Best Clone fitness 1 ', 'Clone generation'])
    # ## save to xlsx file
    # info_filepath = 'info.xlsx'
    # pre_filepath = 'precision_objects.npy'
    # fit_filepath = 'fitness_objects.npy'
    #
    # info_df.to_excel(info_filepath, index=False, header=True)
    # np.save(pre_filepath, pre_data)
    # np.save(fit_filepath, fit_data)