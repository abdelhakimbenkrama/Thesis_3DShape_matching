import os
import pandas as pd
import numpy as np

files_link = "/mnt/c/Users/NEW.PC/Desktop/exelfiles"
files_list = os.listdir(files_link)

all_data = []
for file in files_list:
    data = pd.read_excel(os.path.join(files_link, file)).values
    for row in data:
        all_data.append(row)
dataset_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
                 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
all_data = np.asarray(all_data)
results = []
for i in range(10):
    count = 0
    original_pre =0
    clone_pre = 0
    original_fit =0
    clone_fit = 0
    for row in all_data:
        if row[1] == i :
            original_pre += row[2]
            clone_pre += row[3]
            original_fit += row[5]
            clone_fit += row[6]
            count +=1
    results.append([dataset_names[i] ,original_pre /count , clone_pre/count ,
    original_fit /count, clone_fit/count ])

results = np.asarray(results)

info_df = pd.DataFrame(results, columns=['Class', 'LSD precision', 'MAS precision', 'LSD fitness', 'MAS fitness'])
## save to xlsx file
info_filepath = '/mnt/c/Users/NEW.PC/Desktop/Results.xlsx'
info_df.to_excel(info_filepath, index=False, header=True)