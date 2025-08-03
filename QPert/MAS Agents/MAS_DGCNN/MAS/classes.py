from MAS_DGCNN.MAS.Helper import Two_Vectors_CrosseOver_Mutation, norm_matrix
from MAS_DGCNN.MAS.DGCNN import DGCNN , Identity , vector_maker ,multi_vector_maker
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


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
            dist = np.linalg.norm(self.query - element[:1024])
            distances.append((element[:1024], dist, element[1024]))
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
        disatnces =  self.M_Qi[: , 1]
        for e in disatnces:
            total += e
        self.fitness = total / self.Query_cmp_limits


class CoordinateAgent:
    def __init__(self, object, classe):
        self.Object3d = np.expand_dims(object, axis=0)  # done
        self.classe = classe  # done
        self.model = None  # done
        self.Clones = 0  # done
        self.Discriptors = None  # each generation Descriptors --> new gen
        self.generation = 0
        self.Object_discriptor = None
        self.N = None
        self.object_precision_fitness_data = None
        self.best_precision_clone_data = None
        self.best_fitness_clone_data = None

    def creat_clones(self, n):
        self.N = n
        # TODO controle the shape of 3d object because we may pass a batch of data
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

    def loadModel(self,weights_path):
        # loading model
        model = DGCNN({})
        #model = nn.DataParallel(model)
        model.load_state_dict(torch.jit.load(weights_path, map_location=torch.device('cpu')))
        model.conv5 = Identity()
        model.linear1 = Identity()
        model.bn6 = Identity()
        model.dp1 = Identity()
        model.linear2 = Identity()
        model.bn7 = Identity()
        model.dp2 = Identity()
        model.linear3 = Identity()
        self.model = model.eval()

    def discriptor_maker_object(self):
        CP = np.squeeze(self.Object3d)
        Clout_points = np.expand_dims(CP, axis=0)
        Clout_points = torch.from_numpy(Clout_points)
        Clout_points = Clout_points.permute(0, 2, 1)
        output = self.model(Clout_points)
        output = output.detach().numpy()
        score =  output.squeeze()
        norm = np.linalg.norm(score)
        self.Object_discriptor = score / norm

    def discriptor_maker_clones(self):
        Clones = np.float32(self.Clones)
        Clones_Data_Loader = DataLoader(Clones, batch_size=10, drop_last=False)
        for Clout_points in Clones_Data_Loader:
            Clout_points = Clout_points.permute(0, 2, 1)
            output = self.model(Clout_points)
            output = output.detach().numpy()
            res =  output.squeeze()
        # for Clone in Clones:
        #     CP = np.squeeze(Clone)
        #     res = vector_maker(CP, weights_path)
        #     des.append(res)
        #des = np.asarray(des)
        #normalize the descriptors vectors
        self.Discriptors = norm_matrix(res)

    def CrosseOver_Mutation(self):
        Data = self.Clones
        lenght = Data.shape[0]
        # new_gen = []
        new_gen = np.zeros((10, 2048, 3))
        indexs = np.random.choice(range(lenght), lenght, replace=False)
        for x in range(0, lenght, 2):
            # x_, y_ = Two_Vectors_CrosseOver_Mutation(Data[indexs[x]], Data[indexs[x + 1]])
            new_gen[indexs[x]], new_gen[indexs[x + 1]] = Two_Vectors_CrosseOver_Mutation(Data[indexs[x]],
                                                                                         Data[indexs[x + 1]])
        #     new_gen.append(x_)
        #     new_gen.append(y_)
        # self.Clones = np.asarray(new_gen)
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
