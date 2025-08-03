import numpy as np
from helper import *

class RetrievalAgent:
    def __init__(self, query, label, Query_cmp_limits):
        self.Query_cmp_limits = Query_cmp_limits
        self.query = query
        self.label = label
        self.M_Qi = 0
        self.precision = 0
        self.fitness = 0

    # helper function
    def trns(self, arr):
        res = []
        for elm in arr:
            res.append([elm[0], elm[1], elm[2]])
        return np.asarray(res)

    # calculate M_Qi
    def oredered_distance_list(self, data_set):
        distances = []
        for element in data_set:
            dist = np.linalg.norm(self.query - element[0])
            distances.append((element[0], dist, element[1]))
        dtype = [('elemnt', object), ('distance', float), ('id', int)]
        data = np.asarray(distances, dtype=dtype)
        data = self.trns(data)
        data = data[np.argsort(data[:, 1])]
        data = data[:self.Query_cmp_limits]
        self.M_Qi = data

    def precision_func(self):
        id = self.label
        nearst_neighbords = self.M_Qi[:, 2]
        positive = 0
        for elm in nearst_neighbords:
            if elm == id:
                positive += 1
        self.precision = positive / self.Query_cmp_limits

    def fitness_func(self):
        total = 0
        for e in self.M_Qi:
            total += e[1]
        self.fitness = total / self.Query_cmp_limits


class CoordinateAgent:
    def __init__(self, obj, classe):
        self.Object3d = np.expand_dims(obj, axis=0)  # done
        self.classe = classe  # done
        self.Clones = 0  # done
        self.Discriptors = None  # each generation Descriptors --> new gen
        self.generation = 0
        self.Object_discriptor = LSD(obj)
        self.N = None
        self.object_precision_fitness_data = None
        self.best_precision_clone_data = None
        self.best_fitness_clone_data = None

    def creat_clones(self, n):
        self.N = n
        Clones = []
        sigma = 0.01
        clip = 0.5
        B, N, C = self.Object3d.shape
        for i in range(n):
            jittered_object = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
            jittered_object += self.Object3d
            Clones.append(jittered_object)
        Clones = np.asarray(Clones)
        self.Clones = np.squeeze(Clones)

    def discriptor_maker_clones(self):
        all_dis = []
        for clone in self.Clones:
            clone_dis = LSD(clone)
            all_dis.append(clone_dis)
        self.Discriptors = np.asarray(all_dis)

    def CrosseOver_Mutation(self):
        Data = self.Clones
        lenght = Data.shape[0]
        new_gen = np.zeros((self.N, 512, 3))
        indexs = np.random.choice(range(lenght), lenght, replace=False)
        for x in range(0, lenght, 2):
            new_gen[indexs[x]], new_gen[indexs[x + 1]] =\
                Two_Vectors_CrosseOver_Mutation(Data[indexs[x]], Data[indexs[x + 1]])
        self.Clones = new_gen

    def save_best_precision_clone(self, result):
        if self.best_precision_clone_data == None:
            self.best_precision_clone_data = result + [self.generation]
        elif self.best_precision_clone_data[1] <= result[1]:
            self.best_precision_clone_data = result + [self.generation]

    def save_best_fitness_clone(self, result):
        if self.best_fitness_clone_data == None:
            self.best_fitness_clone_data = result + [self.generation]
        elif self.best_fitness_clone_data[1] >= result[1]:
            self.best_fitness_clone_data = result + [self.generation]

