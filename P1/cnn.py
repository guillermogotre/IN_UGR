import pandas
from sklearn import neighbors, datasets
import numpy as np
import random

def isCClassified(X,y,r,c):
    print("#",X.shape,y.shape)
    if X.size == 0:
        return False
    clf = neighbors.KNeighborsClassifier(1)
    clf.fit(X,y)
    z = clf.predict(r.reshape(1,-1))
    eq = np.all(z == c)
    return eq

def toNumber(label):
    if label == "popular":
        return 1
    else:
        return 0

def fromNumber(label):
    if label == 1:
        return "popular"
    else:
        return "no_popular"

def CCN(input_table):
    X = np.array([]).reshape((0,len(input_table.columns)-1))
    y = np.array([]).reshape((0,1))

    changed = True
    lst = list(input_table.iterrows())
    while(changed):
        changed = False
        i = 0
        random.shuffle(lst)
        for rid,r in lst:
            i += 1
            r = np.array(r.as_matrix())
            r,c = r[:-1],toNumber(r[-1])
            if not isCClassified(X,y,r,c):
                X = np.vstack((X,r))
                y = np.vstack((y,c))
                changed = True  
        

    res = [list(r) + [fromNumber(l)] for r,l in zip(X,y.reshape((-1)))]
    return res

output_table = CCN(input_table)
