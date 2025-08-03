from dotenv import load_dotenv
from sklearn.cluster import KMeans
import time
import numpy as np
import os
from multiprocessing import Pool
#loading ENV variables
load_dotenv()

model_net_10_link =  os.getenv("ModelNet10Link")
sampled_data_link =  os.getenv('SampledData')
descriptors_saving_link = os.getenv('DescriptorsLink')

dataset = np.load(sampled_data_link , allow_pickle= True)

# calculate distance between a point X and all shape points
def point_shape_distance(point, points):
    point_ = np.full((512, 3), point)
    point_ = np.subtract(point_, points)
    point_ = np.power(point_, 2)
    point_ = np.sum(point_, axis=1)
    point_ = np.sqrt(point_)
    return point_[(0.3 > point_)]

# multi processing function to creat ModelNet lsd discriptors
def processing(index):
    n = 512
    points = dataset[index][0]
    label = dataset[index][1]
    # creat a list of histograms
    hists = []
    #normalize the points
    norm = np.linalg.norm(points)
    points = points/norm
    # calculate distances between each choosen points and all shape points
    for i in range(n):
        distance = point_shape_distance(points[i], points)
        bins, hist = np.histogram(distance, bins=31)
        hists.append(hist)
    hists = np.asarray(hists)
    # get K Histograms using K-means
    kmeans = KMeans(n_clusters=40, random_state=0).fit(hists)
    centers = kmeans.cluster_centers_
    return ([centers , label])

if __name__ == '__main__':
    t1 = time.time()
    p = Pool()
    results = p.map(processing, range(len(dataset)))
    p.close()
    p.join()
    print("processing  : ", time.time() - t1)
    # Save results
    res = np.array(results, dtype="O")
    print(res.shape)
    np.save(descriptors_saving_link, res)