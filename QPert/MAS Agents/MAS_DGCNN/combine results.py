import os
# load files names
path = "C:\\Users\\NEW.PC\\Desktop\\dgcnn_results\\results\\Exel"
files_list = os.listdir(path)
print(len(files_list))
for x in files_list:
    print(x)
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


