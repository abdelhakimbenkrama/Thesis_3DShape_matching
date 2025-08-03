import numpy as np
#raw_data: all histograms
#clustering: indices of cluster the point belonging to
#k: the number of clusters


def display(raw_data, clustering, k):

  clst = []
  (n, dim) = raw_data.shape
  #print (n, " dim ", dim)
  #print("Clustering of n histograms (one per point on the shape)")
  #print("-------------------")
  for kk in range(k):  # group by cluster ID (for each cluster)
    hist_by_cluster = [0] * dim
    nb_elem_by_cluster = 0
    for i in range(n):  # scan the raw data (the index number of the point)
      c_id = clustering[i]  # cluster ID of curr item
      if c_id == kk:
        #print("%4d " % i, end="");
        #print(raw_data[i])
        hist_by_cluster = hist_by_cluster + raw_data[i]
        nb_elem_by_cluster =nb_elem_by_cluster +1
    if nb_elem_by_cluster != 0:
        hist_by_cluster =hist_by_cluster / nb_elem_by_cluster
    clst.append(hist_by_cluster)
    #print("size of the cluster ", nb_elem_by_cluster)
    #print("The hist ", kk ," is \n" , hist_by_cluster)
    #print("-------------------")
  return  clst

def distance(item, mean):
  sum = 0.0
  dim = len(item)
  for j in range(dim):
    sum += (item[j] - mean[j]) ** 2
  return np.sqrt(sum)

def initialize(norm_data, k):
  (n, dim) = norm_data.shape
  clustering = np.zeros(shape=(n), dtype=np.int)
  for i in range(k):
    clustering[i] = i
  for i in range(k, n):
    clustering[i] = np.random.randint(0, k)

  means = np.zeros(shape=(k,dim), dtype=np.float32)
  update_means(norm_data, clustering, means)
  return(clustering, means)

def mm_normalize(data):
  (rows, cols) = data.shape  # (20,4)
  mins = np.zeros(shape=(cols), dtype=np.float32)
  maxs = np.zeros(shape=(cols), dtype=np.float32)
  for j in range(cols):
    mins[j] = np.min(data[:,j])
    maxs[j] = np.max(data[:,j])

  result = np.copy(data)
  for i in range(rows):
    for j in range(cols):
      result[i,j] = (data[i,j] - mins[j]) / (maxs[j] - mins[j])
  return (result, mins, maxs)

def update_means(norm_data, clustering, means):
  # given a (new) clustering, compute new means
  # assumes update_clustering has just been called
  # to guarantee no 0-count clusters
  (n, dim) = norm_data.shape
  k = len(means)
  counts = np.zeros(shape=(k), dtype=np.int)
  new_means = np.zeros(shape=means.shape, dtype=np.float32)  # k x dim
  for i in range(n):  # walk thru each data item
    c_id = clustering[i]
    counts[c_id] += 1
    for j in range(dim):
      new_means[c_id,j] += norm_data[i,j]  # accumulate sum

  for kk in range(k):  # each mean
    for j in range(dim):
      new_means[kk,j] /= counts[kk]  # assumes not zero

  for kk in range(k):  # each mean
    for j in range(dim):
      means[kk,j] = new_means[kk,j]  # update by ref

def update_clustering(norm_data, clustering, means):
    # given a (new) set of means, assign new clustering
    # return False if no change or bad clustering
    n = len(norm_data)
    k = len(means)

    new_clustering = np.copy(clustering)  # proposed clustering
    distances = np.zeros(shape=(k), dtype=np.float32)

    for i in range(n):  # walk thru each data item
        for kk in range(k):
            distances[kk] = distance(norm_data[i], means[kk])
        new_id = np.argmin(distances)
        new_clustering[i] = new_id

    if np.array_equal(clustering, new_clustering):  # no change
        return False

    # make sure that no cluster counts have gone to zero
    counts = np.zeros(shape=(k), dtype=np.int)
    for i in range(n):
        c_id = clustering[i]
        counts[c_id] += 1

    for kk in range(k):  # could use np.count_nonzero
        if counts[kk] == 0:  # bad clustering
            return False

    # there was a change, and no counts have gone 0
    for i in range(n):
        clustering[i] = new_clustering[i]  # update by ref
    return True

def cluster(norm_data, k):
  (clustering, means) = initialize(norm_data, k)
  ok = True  # if a change was made and no bad clustering
  max_iter = 100
  sanity_ct = 1
  while sanity_ct <= max_iter:
    ok = update_clustering(norm_data, clustering, means)
    if ok == False:
      break  # done
    update_means(norm_data, clustering, means)
    sanity_ct += 1
  return clustering
