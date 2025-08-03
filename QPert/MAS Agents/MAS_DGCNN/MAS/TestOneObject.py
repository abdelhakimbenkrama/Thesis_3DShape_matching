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
path_to_discriptors = "S:\\Phd Work\\MAS work\\data\\DGCNN_descriptors\\"
objects_Discriptors_data = load_discriptors(path_to_discriptors)

# path to the pretrained weights of pointnet model
path_to_weights = "S:\\Phd Work\\MAS work\\data\\DGCNN_descriptors\\Dgcnn_weights_2048_50_epochs.t7"


Element_per_class = [725, 156, 615, 193, 672, 435, 84, 297, 989, 187,
                     99, 157, 286, 129, 286, 169, 271, 255, 165, 144,
                     169, 384, 565, 286, 108, 331, 339, 124, 215, 148,
                     780, 144, 110, 492, 183, 444, 367, 575, 107, 123]

Element_per_class = np.asarray(Element_per_class)

inde = []
MYCLASS = 3
for i , lab in enumerate(y) :
    if int(lab) == MYCLASS:
        inde.append(i)

# PARAMETERS
N = 10
GENERATIONS = 20


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

    print("Number of Classe Object : ",R_agent_for_original_obj.Query_cmp_limits)
    print("Fitness : ",R_agent_for_original_obj.fitness)
    print("Precision : ",R_agent_for_original_obj.precision)

processing(inde[0])