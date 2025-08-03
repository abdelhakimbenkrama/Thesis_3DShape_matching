import pandas as pd
import random
def shrink_indexs(mylist, parts):
    list_len = len(mylist)
    batch = int(list_len / parts)
    myindexs = []
    index =1
    start = 0
    end = batch
    while index < parts :
        myindexs.append(mylist[start:end])
        start += batch
        end += batch
        index +=1
    myindexs.append(mylist[start:])
    return myindexs

