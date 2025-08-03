# import verctors and data set , diffrent Agents
import numpy as np
from Mas_PoinNet.MAS.Helper import Load_H5_Dataset ,Load_PointNet_Discriptors
from Mas_PoinNet.MAS.classes import CoordinateAgent as PAgent
from MAS_DGCNN.MAS.Helper import load_discriptors
from MAS_DGCNN.MAS.classes import CoordinateAgent as DAgent
import time
from tqdm import tqdm

Path_To_Pointnet_Weights = "S:\\Phd Work\\MAS work\\data\\pointNet_weights.h5"
path_to_DGCNN_discriptors = "S:\\Phd Work\\MAS work\\data\\DGCNN_descriptors\\"
path_to_weights = "S:\\Phd Work\\MAS work\\data\\model.2048.t7"
#PointNet
x, y =Load_H5_Dataset()
PointNetDiscriptors  = Load_PointNet_Discriptors()

print(type(PointNetDiscriptors[0][1024]))
DGCnnDiscriptors= load_discriptors(path_to_DGCNN_discriptors)
DGCnnDiscriptors = DGCnnDiscriptors.astype(np.float32)
print(type(DGCnnDiscriptors[0][1024]))
# time to create 10 CLones Discriptor
def PointNetClonesDiscriptors():
    t1 = time.time()
    C_agent = PAgent(x[0] ,y[0])
    C_agent.get_Point_Net_Model(Path_To_Pointnet_Weights)
    C_agent.creat_clones(10)
    C_agent.discriptor_maker_clones()
    print("time to create 10 Clones Discriptors" ,time.time() - t1)


# time to calculate Distance between 2 vectors
def PoitnNettwoVectorsTime():
    t1 = time.time()
    C_agent = PAgent(x[0], y[0])
    C_agent.get_Point_Net_Model(Path_To_Pointnet_Weights)
    C_agent.discriptor_maker_object()
    object = PointNetDiscriptors[0]
    t1 = time.time()
    dist = np.linalg.norm(C_agent.Object_discriptor - object[:1024])
    print("2 vectors Distance is Calculated in : " , time.time() -t1)

# time to calculate distance between object and dataset offline
def PointNetoffline_MQi():
    C_agent = PAgent(x[0], y[0])
    C_agent.get_Point_Net_Model(Path_To_Pointnet_Weights)
    C_agent.creat_clones(10)
    C_agent.discriptor_maker_clones()
    t1 = time.time()
    Clone = C_agent.Discriptors[0]
    distances =[]
    for element in tqdm(PointNetDiscriptors) :
        dist = np.linalg.norm(Clone - element[:1024])
        distances.append((element[:1024], dist, element[1024]))
    print("offline MQi took: " , time.time() - t1 )

#DGCNN

def DGCNNClonedescriptor():
    t1 = time.time()
    C_agent = DAgent(x[0] ,y[0])
    C_agent.creat_clones(10)
    C_agent.loadModel(path_to_weights)
    C_agent.discriptor_maker_clones()
    print("Processing Time : ",time.time() - t1)

def DGCnnTwoVectorsTime():
    t1 = time.time()
    C_agent = DAgent(x[0], y[0])
    C_agent.loadModel(path_to_weights)
    C_agent.discriptor_maker_object()
    object = DGCnnDiscriptors[0]
    dist = np.linalg.norm(C_agent.Object_discriptor - object[:1024])
    print("2 vectors Distance is Calculated in : ", time.time() - t1)

def DGCnnOffline_MQi():
    C_agent = DAgent(x[0], y[0])
    C_agent.creat_clones(10)
    C_agent.loadModel(path_to_weights)
    C_agent.discriptor_maker_clones()
    t1 = time.time()
    Clone = C_agent.Discriptors[0]
    Clone = Clone.astype(np.float32)
    distances = []
    for element in tqdm(DGCnnDiscriptors):
        dist = np.linalg.norm(Clone - element[:1024])
        distances.append((element[:1024], dist, element[1024]))
    print("offline MQi took: ", time.time() - t1)
