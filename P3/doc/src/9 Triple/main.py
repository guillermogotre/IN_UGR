import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

def openDataset():
    '''
    lectura de datos
    '''
    # los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
    data_x = pd.read_csv('data/water_pump_tra.csv')
    data_y = pd.read_csv('data/water_pump_tra_target.csv')
    data_x_tst = pd.read_csv('data/water_pump_tst.csv')

    # se quitan las columnas que no se usan
    data_x.drop(labels=['id'], axis=1, inplace=True)
    #data_x.drop(labels=['date_recorded'], axis=1, inplace=True)

    data_x_tst.drop(labels=['id'], axis=1, inplace=True)
    #data_x_tst.drop(labels=['date_recorded'], axis=1, inplace=True)

    data_y.drop(labels=['id'], axis=1, inplace=True)

    return data_x, data_y, data_x_tst

import datetime
def parsed(x):
    TIME_STR = '%Y-%m-%d'
    return datetime.datetime.strptime(x, TIME_STR)

def getDays(date_c,min_date=parsed('2002-10-14')):
    if min_date is None:
        min_date = parsed(date_c.min())
    days = [(parsed(d) - min_date).days for d in date_c]
    return days

def getMonth(date_c):
    months = [parsed(d).month for d in date_c]
    return months

def addDatesPrep(data):
    days = getDays(data['date_recorded'])
    data['days_ellapsed'] = days
    months = getMonth(data['date_recorded'])
    data['month_n'] = months

from geopy.distance import geodesic
def addGeoDist(data):
    coords = list(zip(data['longitude'],data['latitude']))
    msk = np.abs(np.sum(coords,axis=1)) < 1e-5
    data['reliable_gps'] = ['T' if m else 'F' for m in msk]
    kms = [geodesic(p1,(0,0)).km for p1 in coords]
    data['dist_ori'] = kms
    return msk

from sklearn.neighbors.classification import KNeighborsClassifier
from collections import Counter

GPS_KEYS = ["basin", "subvillage", "region", "region_code", "district_code", "lga", "ward"]
CAT_KEYS = ["funder","installer","wpt_name","basin","subvillage","region","lga","ward","recorded_by","scheme_management","scheme_name","extraction_type","extraction_type_group","extraction_type_class","management","management_group","payment","payment_type","source","source_type","source_class","water_quality","quality_group","quantity","quantity_group","waterpoint_type","waterpoint_type_group"]
BIG_CAT_KEYS = ['funder', 'installer', 'wpt_name', 'subvillage', 'ward', 'scheme_name']
SMALL_CAT_KEYS = ['basin', 'region', 'lga', 'scheme_management', 'extraction_type', 'extraction_type_group', 'extraction_type_class', 'management', 'management_group', 'payment_type', 'source', 'source_type', 'source_class', 'water_quality', 'quantity', 'waterpoint_type', 'waterpoint_type_group']
INTERSTING_SMALL = ['basin','lga','payment_type', 'source', 'waterpoint_type']
TODROP_KEYS = ["recorded_by","payment","quality_group","quantity_group"]
import multiprocessing.dummy as mp

def replaceGps(ref,ref_y,dst):
    #gGps = data[np.logical_not(msk)][KEYS].values
    gGps = ref
    #gY = data[np.logical_not(msk)][['longitude','latitude']].values
    gY = ref_y
    #bGps = data[msk][KEYS].values
    bGps = dst


    dist = lambda x1,x2: np.sum([l1==l2 for l1,l2 in zip(x1,x2)])

    KNN = 3
    res = []

    def knn(row):
        d = np.apply_along_axis(lambda x: dist(x, row), 1, gGps)
        m = np.hstack((d.reshape((-1, 1)), gY))
        m = sorted(m, key=lambda x: x[0], reverse=True)
        mnn = np.array(m[:KNN])
        mnn[:, 1] *= mnn[:, 0]
        mnn[:, 2] *= mnn[:, 0]
        #res.append(np.mean(mnn, axis=0)[1:] / np.mean(mnn, axis=0)[0])
        #if mnn[0,0] == 0:
        #    print("ZERO")
        #    return [0,0]
        #else:
        return np.mean(mnn, axis=0)[1:] / np.mean(mnn, axis=0)[0]

    p = mp.Pool(4)
    res = p.map(knn, bGps)
    p.close()
    p.join()
    # for row in bGps:
    #     d = np.apply_along_axis(lambda x: dist(x,row),1,gGps)
    #     m = np.hstack((d.reshape((-1,1)),gY))
    #     m = sorted(m,key=lambda x: x[0],reverse=True)
    #     mnn = np.array(m[:KNN])
    #     mnn[:, 1] *= mnn[:, 0]
    #     mnn[:, 2] *= mnn[:, 0]
    #     res.append(np.mean(mnn,axis=0)[1:]/np.mean(mnn,axis=0)[0])

    #a = 0
    #data[msk][['longitude','latitude']] = np.array(res)
    return res


import os.path
import copy
from gensim.models import Word2Vec
from random import shuffle


# https://www.kaggle.com/classtag/cat2vec-powerful-feature-for-categorical

def getW2V2(data_x,data_x_tst):
    daset =  pd.concat([data_x,data_x_tst],axis=0)

    def apply_w2v(sentences, model, num_features):
        def _average_word_vectors(words, model, vocabulary, num_features):
            feature_vector = np.zeros((num_features,), dtype="float64")
            n_words = 0.
            for word in words:
                if word in vocabulary:
                    n_words = n_words + 1.
                    feature_vector = np.add(feature_vector, model[word])

            if n_words:
                feature_vector = np.divide(feature_vector, n_words)
            return feature_vector

        vocab = set(model.wv.index2word)
        feats = [_average_word_vectors(s, model, vocab, num_features) for s in sentences]
        return np.array(feats)

    def gen_cat2vec_sentences(data):
        X_w2v = copy.deepcopy(data)
        names = list(X_w2v.columns.values)
        for c in names:
            X_w2v[c] = X_w2v[c].fillna('unknown').astype('category')
            X_w2v[c].cat.categories = ["%s %s" % (c, g) for g in X_w2v[c].cat.categories]
        X_w2v = X_w2v.values.tolist()
        return X_w2v

    n_cat2vec_feature = len(CAT_KEYS)  # define the cat2vecs dimentions
    n_cat2vec_window = len(CAT_KEYS) * 2  # define the w2v window size

    def fit_cat2vec_model():
        X_w2v = gen_cat2vec_sentences(daset.loc[:, CAT_KEYS].sample(frac=0.6))
        for i in X_w2v:
            shuffle(i)
        model = Word2Vec(X_w2v, size=n_cat2vec_feature, window=n_cat2vec_window)
        return model

    c2v_model = fit_cat2vec_model()

    tr_c2v_matrix = apply_w2v(gen_cat2vec_sentences(data_x.loc[:, CAT_KEYS]), c2v_model, n_cat2vec_feature)
    te_c2v_matrix = apply_w2v(gen_cat2vec_sentences(data_x_tst.loc[:, CAT_KEYS]), c2v_model, n_cat2vec_feature)

    data_x.loc[:, CAT_KEYS] = tr_c2v_matrix
    data_x_tst.loc[:, CAT_KEYS] = te_c2v_matrix

    return data_x, data_x_tst

def getW2V(data_x,data_x_tst,size=6,window=12,keys=BIG_CAT_KEYS):
    #size = 6
    #window = 12
    x_w2v = pd.concat([data_x,data_x_tst],axis=0)[keys].copy()
    x_w2v = x_w2v.fillna('unknown')

    names = x_w2v.columns.values

    for i in names:
        x_w2v[i] = x_w2v[i].astype('category')
        x_w2v[i].cat.categories = ["Feature %s %s" % (i,g) for g in x_w2v[i].cat.categories]
    x_w2v = x_w2v.values.tolist()
    for i in x_w2v:
        shuffle(i)
    w2v = Word2Vec(x_w2v,size=size,window=window)

    X_train_w2v = data_x[keys].copy()
    X_test_w2v = data_x_tst[keys].copy()

    for i in names:
        X_train_w2v[i] = X_train_w2v[i].astype('category')
        X_train_w2v[i].cat.categories = ["Feature %s %s" % (i, g) for g in X_train_w2v[i].cat.categories]
    for i in names:
        X_test_w2v[i] = X_test_w2v[i].astype('category')
        X_test_w2v[i].cat.categories = ["Feature %s %s" % (i, g) for g in X_test_w2v[i].cat.categories]
    X_train_w2v = X_train_w2v.values
    X_test_w2v = X_test_w2v.values

    x_w2v_train = np.random.random((len(X_train_w2v), size * X_train_w2v.shape[1]))

    for j in range(X_train_w2v.shape[1]):
        for i in range(X_train_w2v.shape[0]):
            if X_train_w2v[i, j] in w2v:
                x_w2v_train[i, j * size:(j + 1) * size] = w2v[X_train_w2v[i, j]]

    x_w2v_test = np.random.random((len(X_test_w2v), size * X_test_w2v.shape[1]))
    for j in range(X_test_w2v.shape[1]):
        for i in range(X_test_w2v.shape[0]):
            if X_test_w2v[i, j] in w2v:
                x_w2v_test[i, j * size:(j + 1) * size] = w2v[X_test_w2v[i, j]]

    return x_w2v_train, x_w2v_test

def impZeroHeight(src,dst):
    coords_tr = src[['longitude','latitude']]
    height_tr = src[['gps_height']]

    coords_tst = dst[['longitude','latitude']]
    height_tst = dst[['gps_height']]
    tst_msk = (height_tst == 0).values.reshape(-1)
    coords_tst = coords_tst.values[tst_msk]

    msk = (height_tr != 0).values.reshape(-1)
    nonzero_coords = coords_tr.values[msk]
    nonzero_height = height_tr.values[msk]

    rg = lgb.LGBMRegressor(n_estimators=500)
    rg.fit(nonzero_coords, nonzero_height)

    #knn = KNeighborsRegressor(3,'distance')
    #knn.fit(nonzero_coords,nonzero_height)

    lbs = rg.predict(coords_tst)
    dst.iloc[tst_msk, list(map(lambda x: x == 'gps_height', dst.columns))] = lbs.reshape((-1,1))
    a = 0

def reduceBigCat(tr,tst):
    X = pd.concat([tr,tst],axis=0)
    MIN = 50
    gen_label = 'Other'
    for k in list(filter(lambda x: tr[x].dtype == np.dtype('O'), tr.columns)):
        col = X[k].fillna('NaN')
        c = Counter(col.values.reshape(-1))
        msk = col.isin(list(filter(lambda x: c[x] < MIN, c)))
        col[msk] = gen_label
        X[k] = col
    return X.iloc[:tr.shape[0],:], X.iloc[tr.shape[0]:,:]


def impWater(tr,tst):
    X_all = pd.concat([tr, tst], axis=0)
    y_all = X_all['amount_tsh']

    # Ordinate quantity
    qnt = ['dry', 'unknown', 'insufficient', 'seasonal', 'enough']
    for i, k in enumerate(qnt):
        X_all.loc[X_all['quantity'] == k, X_all.columns == 'quantity'] = i

    X = X_all.loc[y_all != 0]
    y = X['amount_tsh']
    X = X[['gps_height','longitude','latitude','dist_ori','reliable_gps','quantity']]

    rg = lgb.LGBMRegressor(n_estimators=500)
    rg.fit(X,y)

    dst = X_all.loc[y_all == 0][['gps_height','longitude','latitude','dist_ori','reliable_gps','quantity']]
    pred = rg.predict(dst)

    X_all.loc[y_all == 0, X_all.columns == 'amount_tsh'] = pred

    tr['amount_tsh'] = X_all['amount_tsh'].values[:tr.shape[0]]
    tst['amount_tsh'] = X_all['amount_tsh'].values[tr.shape[0]:]

    return tr, tst

def ordenateCategories(tr,tst,y):
    keys = list(filter(lambda x: tr[x].dtype == np.dtype('O'), tr.columns))
    for k in keys:
        vals = list(Counter(tr[k].values.reshape(-1)))
        impV = sorted(map(
            lambda x: (np.sum((y[tr[k] == x] == 'functional').values) / np.sum((tr[k] == x).values), x),
            vals),reverse=True)
        d = dict(map(lambda x: (x[1][1], x[0]), enumerate(impV)))

        tr[k] = list(map(lambda x: d.get(x,-1),tr[k].values))
        tst[k] = list(map(lambda x: d.get(x,-1),tst[k].values))

from sklearn.preprocessing import OneHotEncoder
def oneHot(tr,tst):
    keys = INTERSTING_SMALL
    X_all = pd.concat([tr, tst], axis=0)[SMALL_CAT_KEYS]

    oh = OneHotEncoder()
    oh.fit(X_all)
    res = oh.transform(X_all)

    mini_tr = res[:tr.shape[0]].toarray()
    mini_tst = res[tr.shape[0]:].toarray()

    return \
        pd.concat([tr,pd.DataFrame(mini_tr)],axis=1),\
        pd.concat([tst,pd.DataFrame(mini_tst)],axis=1)


def preprocess(data_x, data_y, data_x_tst,w2vsize=6,w2vwindow=12):

    data_x_path = "data_x_gps.pkl"
    data_x_tst_path = "data_x_tst_gps.pkl"
    if os.path.isfile(data_x_path) and os.path.isfile(data_x_tst_path):
        data_x = pd.read_pickle(data_x_path)
        data_x_tst = pd.read_pickle(data_x_tst_path)
    else:
        # Transform Dates
        addDatesPrep(data_x)
        addDatesPrep(data_x_tst)

        # Geo dist
        msk = addGeoDist(data_x)

        data_x_copy = data_x.copy()
        label_mask = [x in GPS_KEYS for x in data_x.columns]
        coord_mask = [x in ['longitude','latitude'] for x in data_x.columns]
        print("Coords begin")
        coords = replaceGps(
            data_x_copy.iloc[np.logical_not(msk),label_mask].values,
            data_x_copy.iloc[np.logical_not(msk),coord_mask].values,
            data_x_copy.iloc[msk,label_mask].values)
        print("Coords 1")
        data_x.iloc[msk, coord_mask] = coords


        msk_2 = addGeoDist(data_x_tst)
        coords = replaceGps(
            data_x_copy.iloc[np.logical_not(msk), label_mask].values,
            data_x_copy.iloc[np.logical_not(msk), coord_mask].values,
            data_x_tst.iloc[msk_2,label_mask].values)
        print("Coords 2")
        data_x_tst.iloc[msk_2, coord_mask] = coords

        data_x.to_pickle(data_x_path)
        data_x_tst.to_pickle(data_x_tst_path)

    data_x_path = "data_x_gps2.pkl"
    data_x_tst_path = "data_x_tst_gps2.pkl"
    if os.path.isfile(data_x_path) and os.path.isfile(data_x_tst_path):
        data_x = pd.read_pickle(data_x_path)
        data_x_tst = pd.read_pickle(data_x_tst_path)
    else:
        impZeroHeight(data_x,data_x)
        impZeroHeight(data_x,data_x_tst)


        # Drop redundant
        data_x.drop(labels=TODROP_KEYS,axis=1,inplace=True)
        data_x_tst.drop(labels=TODROP_KEYS, axis=1, inplace=True)

        # is nan?
        # list(filter(lambda x: x[1]!=0,list(zip(data_x.columns,np.sum(data_x.isnull().values,axis=0)))))
        # l = [('funder', 3662), ('installer', 3696), ('wpt_name', 20), ('subvillage', 371), ('public_meeting', 3334), ('scheme_management', 3877), ('scheme_name', 28174), ('permit', 3056)]

        # Drop date_recorded
        data_x.drop(labels=['date_recorded'], axis=1, inplace=True)
        data_x_tst.drop(labels=['date_recorded'], axis=1, inplace=True)

        # Replace booleans
        #17,21,41
        l = ["public_meeting","permit","reliable_gps"]
        for k in l:
            data_x[k] = [1 if x is True or x=='T' else 0 for x in data_x[k].values]
            data_x_tst[k] = [1 if x is True or x == 'T' else 0 for x in data_x_tst[k].values]

        # Imputate water
        # data_x, data_x_tst = impWater(data_x,data_x_tst)

        # Reduce big
        data_x, data_x_tst = reduceBigCat(data_x,data_x_tst)

        # TODO DATA LEAKING!!!!!
        prev_data_x = data_x.copy()
        ordenateCategories(data_x,data_x_tst,data_y)

        data_x.to_pickle(data_x_path)
        data_x_tst.to_pickle(data_x_tst_path)


    data_x, data_x_tst = oneHot(data_x,data_x_tst)

    a = 0


    # # To decode
    # o_msk = [data_x[k].dtype == np.dtype('O') for k in data_x.columns]
    # e_msk = np.logical_not(o_msk)
    #
    # # Join all
    # all_data = pd.concat([data_x,data_x_tst],axis=0)
    # all_data_o = all_data.iloc[:,o_msk]
    #
    # all_data_small = all_data_o.iloc[:,list(map(lambda  x: x in SMALL_CAT_KEYS, all_data_o.columns))]
    # all_data_big = all_data_o.iloc[:, list(map(lambda x: x in BIG_CAT_KEYS, all_data_o.columns))]
    #
    # a = 0
    #
    #
    # # Label encoder
    # le = preprocessing.LabelEncoder()
    # mask = all_data_o.isnull()
    # data_x_tmp = all_data_o.fillna(9999)
    # data_x_tmp = data_x_tmp.astype(str).apply(le.fit_transform)
    # data_x_nan = data_x_tmp.where(~mask, all_data_o)
    #
    # # Split tr
    # data_x_tr_o = data_x_nan.iloc[:data_x.shape[0], :]
    # data_x_tr_e = data_x.iloc[:,e_msk]
    # data_x_tr = pd.concat([data_x_tr_o,data_x_tr_e],axis=1)
    #
    # # Split tst
    # data_x_tst_o = data_x_nan.iloc[data_x.shape[0]:, :]
    # data_x_tst_e = data_x_tst.iloc[:,e_msk]
    # data_x_tst = pd.concat([data_x_tst_o, data_x_tst_e], axis=1)
    # a = 0


    # Few variables SMALL_CAT_KEYS

    # Lot of variables BIG_CAT_KEYS


    # Label encoder
    # le = preprocessing.LabelEncoder()
    # mask = data_x.isnull()
    # data_x_tmp = data_x.fillna(9999)
    # data_x_tmp = data_x_tmp.astype(str).apply(le.fit_transform)
    # data_x_nan = data_x_tmp.where(~mask, data_x)
    #
    # mask = data_x_tst.isnull()  # máscara para luego recuperar los NaN
    # data_x_tmp = data_x_tst.fillna(9999)  # LabelEncoder no funciona con NaN, se asigna un valor no usado
    # data_x_tmp = data_x_tmp.astype(str).apply(le.fit_transform)  # se convierten categóricas en numéricas
    # data_x_tst_nan = data_x_tmp.where(~mask, data_x_tst)  # se recuperan los NaN



    X = data_x.values
    X_tst = data_x_tst.values
    y = np.ravel(data_y.values)

    # # Without categorical 0.70351848 0.72959596
    # X = np.hstack((data_x.values,tr_w2v))
    # X_tst = np.hstack((data_x_tst.values,tst_w2v))

    y = np.ravel(data_y.values)

    return X, X_tst, y, data_x.columns

def validacion_cruzada(modelo, X, y, cv, print):
    y_test_all = []
    v_acc = []
    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        acc = accuracy_score(y[test],y_pred)
        v_acc += [acc]
        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(acc, tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all, v_acc

class CustomClassifier:
    clfs = []
    def __init__(self,*clf):
        self.clfs = clf


    def fit(self,X,y):
        for clf in self.clfs:
            clf.fit(X,y)
        return self

    def predict(self,X):
        #TODO assert same class
        res = [clf.predict_proba(X) for clf in self.clfs]
        k = self.clfs[0].classes_
        z = np.zeros(res[0].shape)

        for r in res:
            z += r

        pos = np.apply_along_axis(lambda x: np.where(x == x.max()), axis=1, arr=z).reshape(-1)
        return list(map(lambda i: k[i], pos))



def train(X,y,n_estimators=200, do_print=True, col_names = None):
    if not do_print:
        cprint = lambda x: ""
    else:
        cprint = print
    #print("------ XGB...")
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
    clf1 = xgb.XGBClassifier(n_estimators = 500, max_depth=14, n_jobs=8, objective='multi:softmax')
    # clf, y_test_clf, v_acc = validacion_cruzada(clf,X,y,skf, cprint)
    # # #

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
    print("------ LightGBM...")
    clf2 = lgb.LGBMClassifier(objective='multiclass', n_estimators=1000, n_jobs=-1)
    from sklearn.ensemble import RandomForestClassifier
    clf3 = RandomForestClassifier(n_estimators=500,max_depth=20,n_jobs=-1)

    clf = CustomClassifier(clf1,clf2,clf3)
    #clf = clf1


    lgbm, y_test_lgbm, v_acc = validacion_cruzada(clf, X, y, skf, cprint)
    #sorted(list(zip(clf.feature_importances_,range(231))),reverse=True)

    #clf = clf.fit(X, y, feature_name=list(col_names))
    clf = clf.fit(X, y)
    # lgb.plot_importance(clf)
    # plt.show()
    y_pred_tra = clf.predict(X)
    y_pred_acc = accuracy_score(y, y_pred_tra)
    #print("Score: {:.4f}".format(y_pred_acc))

    print("LGBM, 5-fold: {:.4f}/{:.4f} [{}]".format(
        np.average(v_acc),
        y_pred_acc,
        ",".join(["{:.4f}".format(v) for v in v_acc])
    )
    )

    return clf, {
            'cv_acc_mean': np.average(v_acc),
            'cv_acc': v_acc,
            'x_acc': y_pred_acc
        }

def test(clf,X_tst):
    y_pred_tst = clf.predict(X_tst)
    return y_pred_tst

def savePrediction(y_pred_tst):
    df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
    df_submission['status_group'] = y_pred_tst
    df_submission.to_csv("submission.csv", index=False)

'''
cat_key,dif_values
funder,1897
installer,2144
wpt_name,37400
basin,9
subvillage,19288
region,21
lga,125
ward,2092
recorded_by,1
scheme_management,13
scheme_name,2696
extraction_type,18
extraction_type_group,13
extraction_type_class,7
management,12
management_group,5
payment,7
payment_type,7
source,10
source_type,7
source_class,3
water_quality,8
quality_group,6
quantity,5
quantity_group,5
waterpoint_type,7
waterpoint_type_group,6
'''
def main():
    data_x, data_y, data_x_tst = openDataset()
    #TODO KNN normalize
    X,X_tst,y, col_names = preprocess(data_x,data_y,data_x_tst,7,11)
    clf, cv_acc = train(X, y, 1000, True, col_names)
    y_pred_tst = test(clf, X_tst)
    savePrediction(y_pred_tst)

    # N = 10
    # m = np.zeros((N, 4))
    # for i in range(N):
    #     s = np.random.randint(1,24)
    #     w = np.random.randint(3,24)
    #     X,X_tst,y = preprocess(data_x,data_y,data_x_tst,s,w)
    #     clf, cv_acc = train(X, y,200,do_print=False)
    #     m[i] = [s,w,cv_acc['cv_acc_mean'],cv_acc['x_acc']]
    #
    # print(m)
    # X,X_tst,y = preprocess(data_x, data_y, data_x_tst)
    # m = np.zeros((5,2))
    # '''
    # [[1.00000000e+02 7.88468012e-01]
    #  [2.00000000e+02 7.98063929e-01]
    #  [4.00000000e+02 8.04932542e-01]
    #  [8.00000000e+02 8.07912374e-01]
    #  [1.20000000e+03 8.07323163e-01]]
    # '''
    # # for idx,n in enumerate([700,850,1000,1150]):
    # #     clf, cv_acc = train(X,y,n,False)
    # #     #print("{}\t{}".format(i,cv_acc['cv_acc_mean']))
    # #     m[idx] = [n,cv_acc['cv_acc_mean']]
    # clf, cv_acc = train(X, y)
    # #print( m)
    # y_pred_tst = test(clf,X_tst)
    # savePrediction(y_pred_tst)


if __name__ == "__main__":
    main()