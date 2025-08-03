from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
import numpy as np


def normalize_vector(an_array):  # Normaizing a given array
    norm = np.linalg.norm(an_array)
    normal_array = an_array / norm
    return normal_array


def x2_distance(hist1, hist2):  # distance between 2 histograms
    distance = wasserstein_distance(hist1, hist2)
    return distance


def optimal_hungarian(hists1, hists2):  # Linear sum assignement (Hangarian method)
    # used to calculate distances between two objects represented each one by a set of histograms
    # -> distnce between two objetc
    distance = 0
    # calculate the matrix distance
    matrix = []
    for h1 in hists1:
        sublist = []
        for h2 in hists2:
            sublist.append(x2_distance(h1, h2))
        matrix.append(sublist)
    mat = np.array(matrix)
    mat = np.nan_to_num(mat)
    # print("The cost matrix: ", mat)
    row_ind, col_ind = linear_sum_assignment(mat, maximize=False)
    distance = mat[row_ind, col_ind].sum()
    # print("The distance between 2 LSD-based objects= " , distance)
    return distance


#laod histograms
#
# histograms = np.load("C:\\Users\\NEW.PC\Desktop\multi_pro\MAS_LSD\LSD\modelnet_lsd_discriptors.npy" , allow_pickle=True)
# print(histograms.shape)
# print(histograms[0][1])
# import time
# t0 = time.time()
# distances = []
# # for hist in histograms:
#     # distances.append(np.linalg.norm(histograms[0][1] - hist[1]))
#     # distances.append(wasserstein_distance(histograms[0][1] - hist[1]))
#     # distances.append(optimal_hungarian(histograms[0][1] , hist[1]))
# # print(optimal_hungarian(histograms[0][1] , histograms[0][1]))
# # print(np.linalg.norm(histograms[0][1] - histograms[0][1]))
# print(np.asarray(distances).shape)
# print("execution time : " , time.time() - t0)