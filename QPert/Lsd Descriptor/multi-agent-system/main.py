from classes import *
import os
from multiprocessing import Pool
import time
import numpy as np
from dotenv import load_dotenv
import warnings
import pandas as pd
import multiprocessing

warnings.filterwarnings("ignore")

load_dotenv()

sampled_data_link = os.getenv('SampledDataPc')
lsd_discriptor_link = os.getenv('DescriptorsLinkPc')

dataset = np.load(sampled_data_link, allow_pickle=True)  # shape n * (512 ,3 )
lsd_descriptors = np.load(lsd_discriptor_link, allow_pickle=True)  # shape * (40 ,32 )

Element_per_class_modelnet_40 = [725, 156, 615, 193, 672, 435, 84, 297, 989, 187,
                                 99, 157, 286, 129, 286, 169, 271, 255, 165, 144,
                                 169, 384, 565, 286, 108, 331, 339, 124, 215, 148,
                                 780, 144, 110, 492, 183, 444, 367, 575, 107, 123]

Element_per_class_modelnet_10 = [156, 615, 989, 286, 286, 565, 286, 780, 492, 444]
Element_per_class = np.asarray(Element_per_class_modelnet_10)
N = 10
GENERATIONS = 1

# results
results = []


def processing(ele_index):
    print(ele_index)
    label = dataset[ele_index][1]
    object_data = dataset[ele_index][0]
    # create cordination Agent
    Query_cmp_limits = int(np.squeeze(Element_per_class[label]))
    C_agent = CoordinateAgent(object_data, label)
    C_agent.creat_clones(N)
    C_agent.discriptor_maker_clones()

    # Original Object Fitness
    R_agent_for_original_obj = RetrievalAgent(C_agent.Object_discriptor, C_agent.classe, Query_cmp_limits)
    R_agent_for_original_obj.oredered_distance_list(lsd_descriptors)
    R_agent_for_original_obj.precision_func()
    R_agent_for_original_obj.fitness_func()
    C_agent.object_precision_fitness_data = [R_agent_for_original_obj.query, R_agent_for_original_obj.precision,
                                             R_agent_for_original_obj.fitness]
    # THIS PART SHOULD EXECUTED AS EVENTS
    for i in range(GENERATIONS):
        for j, dis in enumerate(C_agent.Discriptors):
            R_agent = RetrievalAgent(dis, C_agent.classe, Query_cmp_limits)
            R_agent.oredered_distance_list(lsd_descriptors)
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

    return ([
        int(ele_index), int(label),
        float("{:.6f}".format(C_agent.object_precision_fitness_data[1])),
        float("{:.6f}".format(C_agent.best_precision_clone_data[1])),
        int(C_agent.best_precision_clone_data[2]),
        float("{:.6f}".format(C_agent.object_precision_fitness_data[2])),
        float("{:.6f}".format(C_agent.best_fitness_clone_data[1])),
        int(C_agent.best_fitness_clone_data[2]),
        # np.asarray(C_agent.best_precision_clone_data[0]),
        # np.asarray(C_agent.best_fitness_clone_data[0])
    ])

if __name__ == '__main__':
    t0 = time.time()
    p = Pool(multiprocessing.cpu_count())
    # List of rangers to use :
    # range(0,1000) ,range(1000,2000) ,
    # range(2000,3000) ,range(3000,4000) ,
    # range(4000,4899)
    results = p.map(processing, range(0,1000))
    p.close()
    p.join()
    print("execution time : ", (time.time() - t0) / 60, " min")
    # Save results
    info = np.array(results, dtype="O")

    # print(res.shape)
    print(info.shape)
    # convert your array into a dataframe
    info_df = pd.DataFrame(info,
                           columns=['Object index', 'Object Class',
                                    'Object LSD precision', 'Best Clone precision',
                                    'Clone generation', 'Object LSD fitness 1 ',
                                    'Best Clone fitness 1 ', 'Clone generation'])
    ## save to xlsx file
    info_filepath = 'allclasses_01.xlsx'
    info_df.to_excel(info_filepath, index=False, header=True)
