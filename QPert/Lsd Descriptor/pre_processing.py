import trimesh
import os
from tqdm import tqdm
import numpy as np


# Load all dataset Links
def load_modelnet_links(model_net_link):
    model_net_classes = os.listdir(model_net_link)
    all_links = []
    for index , Class in enumerate(model_net_classes):
        #loop each class training and test objects links
        train_objects_names  = os.listdir(os.path.join(model_net_link,Class,"train"))
        test_objects_names =  os.listdir(os.path.join(model_net_link,Class,"test"))
        for name in train_objects_names:
            if "off" in name :
                all_links.append([os.path.join(model_net_link,Class,"train",name) , index])
        for name in test_objects_names:
            if "off" in name:
                all_links.append([os.path.join(model_net_link,Class,"test",name) , index])
    return all_links

# apply shape sampling
def dataset_sampling(links_list , number_of_cps):
    sampled_data= []
    for link in tqdm(links_list):
        mesh = trimesh.load(link[0])
        points = mesh.sample(number_of_cps)
        sampled_data.append([points , link[1]])
    return sampled_data

# Processing function
def SampledDataMaker(link , saving_link):
    print("Loading Links")
    all_links =  load_modelnet_links(link)
    print("Creating Sampled Data")
    sampled_dataset = dataset_sampling(all_links ,512)
    sampled_dataset = np.asarray(sampled_dataset)
    np.save(saving_link , sampled_dataset)
    print("Processing has Ended")

