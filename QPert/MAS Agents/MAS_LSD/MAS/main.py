from MAS_LSD.MAS.classes import *
import os
from multiprocessing import Pool
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lsd_discriptor_link = "..\LSD\modelnet_lsd_discriptors.npy"

x, y = Load_H5_Dataset()

Element_per_class = [725, 156, 615, 193, 672, 435, 84, 297, 989, 187,
                     99, 157, 286, 129, 286, 169, 271, 255, 165, 144,
                     169, 384, 565, 286, 108, 331, 339, 124, 215, 148,
                     780, 144, 110, 492, 183, 444, 367, 575, 107, 123]
#lsd_discriptors = lsd_discriptors_maker(lsd_discriptor_link, y)

disc = np.load(lsd_discriptor_link,allow_pickle=True)


Element_per_class = np.asarray(Element_per_class)
N = 10
GENERATIONS = 1

# results
results = []


def processing(ele_index):
    # create cordination Agent
    Query_cmp_limits = int(np.squeeze(Element_per_class[y[ele_index]]))
    C_agent = CoordinateAgent(x[ele_index], y[ele_index])
    C_agent.creat_clones(N)
    C_agent.discriptor_maker_clones()

    # Original Object Fitness
    R_agent_for_original_obj = RetrievalAgent(C_agent.Object_discriptor, C_agent.classe, Query_cmp_limits)
    R_agent_for_original_obj.oredered_distance_list(lsd_discriptors)
    R_agent_for_original_obj.precision_func()
    R_agent_for_original_obj.fitness_func()
    C_agent.object_precision_fitness_data = [R_agent_for_original_obj.query, R_agent_for_original_obj.precision,
                                             R_agent_for_original_obj.fitness]
    # THIS PART SHOULD EXECUTED AS EVENTS
    for i in range(GENERATIONS):
        for j, dis in enumerate(C_agent.Discriptors):
            R_agent = RetrievalAgent(dis, C_agent.classe, Query_cmp_limits)
            R_agent.oredered_distance_list(lsd_discriptors)
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


# t1 = time.time()
#
# print("on Generation in " , time.time() - t1)
# if __name__ == '__main__':
#     t0 = time.time()
#     p = Pool()
#     results = p.map(processing, range(4))
#     p.close()
#     p.join()
#     print("execution time : ", (time.time() - t0) / 60, " min")
