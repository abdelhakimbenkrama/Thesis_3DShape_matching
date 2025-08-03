import os
import pandas as pd
import numpy as np

#path = "C:\\Users\\NEW.PC\\Desktop\\results"
path = "C:\\Users\\NEW.PC\\Desktop\\dgcnn_results\\results\\Exel"

files = os.listdir(path)

classes_names = os.listdir("C:\\Users\\NEW.PC\\Desktop\\datasets\\ModelNet40")

def moyen_vector(name, data):
    count = 0
    cnn_pre = 0
    clone_pre = 0
    cnn_fit = 0
    clone_fit = 0
    for d in data:
        count += 1
        cnn_pre += d[1]
        clone_pre += d[2]
        cnn_fit += d[4]
        clone_fit += d[5]
    return classes_names[int(name[-6])], cnn_pre/count, clone_pre/count, cnn_fit/count, clone_fit/count


all_data = []
for file in files:
    data = pd.read_excel(os.path.join(path, file)).values
    all_data.append(moyen_vector(file, data))
print(all_data)
all_data = np.asarray(all_data)

info_df = pd.DataFrame(all_data, columns=['Class', 'CNN precision', 'MAS precision', 'CNN fitness', 'MAS fitness'])
## save to xlsx file
info_filepath = 'Results.xlsx'
info_df.to_excel(info_filepath, index=False, header=True)
