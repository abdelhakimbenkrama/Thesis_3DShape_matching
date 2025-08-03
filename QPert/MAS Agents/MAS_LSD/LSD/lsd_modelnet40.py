from MAS_LSD.LSD.lsd import *
from multiprocessing import Pool

x, y = Load_H5_Dataset()
count = len(x)

np.random.seed(40)

# multi processing function to creat ModelNet discriptors
def processing(index):
    print(index)
    n = 2048
    points = x[index]
    # creat a list of histograms
    hists = []
    # choose random points
    #indexs = np.random.choice(range(2048), n, replace=False)
    # calculate distances between each choosen points and all shape points
    for i in range(2048):
        distance = point_shape_distance(points[i], points)
        bins, hist = np.histogram(distance, bins=31)
        hists.append(hist)
    hists = np.asarray(hists)
    # get K Histograms using K-means
    kmeans = KMeans(n_clusters=40, random_state=0).fit(hists)
    centers = kmeans.cluster_centers_
    return ([index , centers])


results = []
if __name__ == '__main__':
    t1 = time.time()
    p = Pool()
    results = p.map(processing, range(len(x)))
    p.close()
    p.join()
    print("processing  : ", time.time() - t1)
    # Save results
    res = np.array(results, dtype="O")
    print(res.shape)
    filepath = 'modelnet_lsd_discriptors.npy'
    np.save(filepath, res)
