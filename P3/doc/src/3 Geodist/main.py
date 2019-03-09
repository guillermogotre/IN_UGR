import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb

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
    #msk = np.abs(np.sum(coords,axis=1)) < 1e-5
    #data['reliable_gps'] = msk
    kms = [geodesic(p1,(0,0)).km for p1 in coords]
    data['dist_ori'] = kms


def preprocess(data_x, data_y, data_x_tst):
    # Transform Dates
    addDatesPrep(data_x)
    addDatesPrep(data_x_tst)

    # Geo dist
    addGeoDist(data_x)
    addGeoDist(data_x_tst)

    # Label encoder
    le = preprocessing.LabelEncoder()
    mask = data_x.isnull()
    data_x_tmp = data_x.fillna(9999)
    data_x_tmp = data_x_tmp.astype(str).apply(le.fit_transform)
    data_x_nan = data_x_tmp.where(~mask, data_x)

    mask = data_x_tst.isnull()  # máscara para luego recuperar los NaN
    data_x_tmp = data_x_tst.fillna(9999)  # LabelEncoder no funciona con NaN, se asigna un valor no usado
    data_x_tmp = data_x_tmp.astype(str).apply(le.fit_transform)  # se convierten categóricas en numéricas
    data_x_tst_nan = data_x_tmp.where(~mask, data_x_tst)  # se recuperan los NaN

    X = data_x_nan.values
    X_tst = data_x_tst_nan.values
    y = np.ravel(data_y.values)

    return X, X_tst, y

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

def train(X,y,n_estimators=200, do_print=True):
    if not do_print:
        cprint = lambda x: ""
    else:
        cprint = print
    # print("------ XGB...")
    # clf = xgb.XGBClassifier(n_estimators = 200)
    # clf, y_test_clf = validacion_cruzada(clf,X,y,skf)
    #

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
    print("------ LightGBM...")
    clf = lgb.LGBMClassifier(objective='binary', n_estimators=n_estimators, num_threads=4)
    lgbm, y_test_lgbm, v_acc = validacion_cruzada(clf, X, y, skf, cprint)

    clf = clf.fit(X, y)
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

def main():
    data_x, data_y, data_x_tst = openDataset()
    X,X_tst,y = preprocess(data_x, data_y, data_x_tst)
    clf, cv_acc = train(X,y)
    y_pred_tst = test(clf,X_tst)
    savePrediction(y_pred_tst)


if __name__ == "__main__":
    main()