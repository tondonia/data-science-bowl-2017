from __future__ import division
import os.path
import numpy as np
import sys, getopt, argparse
import gzip
import cPickle as pickle
import time
import glob
import datetime
from random import shuffle
import json
import unicodecsv
from random import randint
import ntpath
from collections import defaultdict
import random
import ast


'''
A sample code to train xgboost. 
The OptTrain class takes trainX(features) and trainY(labels) as input and initialise the class for 
training the classifier. The optimize function then trains the classifier and the best model is saved
in 'model_best.pkl'.

A sample code is below:
trainX = np.array([[1,1,1], [2,3,4], [2,3,2], [1,1,1], [2,3,4], [2,3,2]])
trainY = np.array([0,1,0,0,0,1])
a = OptTrain(trainX, trainY)
a.optimize()
'''
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.externals import joblib
from sklearn import model_selection
import numpy as np

class OptTrain:
    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY

        self.level0 = xgb.XGBClassifier(learning_rate=0.325,
                                       silent=True,
                                       objective="binary:logistic",
                                       nthread=-1,
                                       gamma=0.85,
                                       min_child_weight=5,
                                       max_delta_step=1,
                                       subsample=0.85,
                                       colsample_bytree=0.55,
                                       colsample_bylevel=1,
                                       reg_alpha=0.5,
                                       reg_lambda=1,
                                       scale_pos_weight=1,
                                       base_score=0.5,
                                       seed=0,
                                       missing=None,
                                       n_estimators=1920, max_depth=6)
        self.h_param_grid = {'max_depth': hp.quniform('max_depth', 1, 13, 1),
                        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                        'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
                        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
                        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                        'n_estimators': hp.quniform('n_estimators', 10, 200, 5),
                        }
        self.to_int_params = ['n_estimators', 'max_depth']

    def change_to_int(self, params, indexes):
        for index in indexes:
            params[index] = int(params[index])

    # Hyperopt Implementatation
    def score(self, params):
        self.change_to_int(params, self.to_int_params)
        self.level0.set_params(**params)
        score = model_selection.cross_val_score(self.level0, self.trainX, self.trainY, cv=5, n_jobs=-1)
        print('%s ------ Score Mean:%f, Std:%f' % (params, score.mean(), score.std()))
        return {'loss': score.mean(), 'status': STATUS_OK}

    def optimize(self):
        trials = Trials()
        print('Tuning Parameters')
        best = fmin(self.score, self.h_param_grid, algo=tpe.suggest, trials=trials, max_evals=200)

        print('\n\nBest Scoring Value')
        print(best)

        self.change_to_int(best, self.to_int_params)
        self.level0.set_params(**best)
        self.level0.fit(self.trainX, self.trainY)
        joblib.dump(self.level0,'model_best.pkl', compress=True)

def read_labels(labels,filename):
    csvFile = open(filename)
    reader = unicodecsv.DictReader(csvFile,encoding='utf-8')
    for j in reader:
        #if not j['id'] == 'b8bb02d229361a623a4dc57aa0e5c485':
        labels[j['id']] = int(j['cancer'])
    return labels

def read_test_labels(filename):
    labels = []
    csvFile = open(filename)
    reader = unicodecsv.DictReader(csvFile,encoding='utf-8')
    for j in reader:
        #if not j['id'] == 'b8bb02d229361a623a4dc57aa0e5c485':
        labels.append(j['id'])
    return labels



def read_depths(filename):
    depths = {}
    csvFile = open(filename)
    reader = unicodecsv.DictReader(csvFile,encoding='utf-8')
    for j in reader:
        depths[j['key']] = int(j['depth'])
    return depths
    

def create_feature_row(key,num_voxels,fps,depths):
    vals = []
    for fp in fps:
        if not key in fp:
            fp[key] = []
        for idx in range(0,min(num_voxels,len(fp[key]))):
            vals.append(0.0)
            vals.append(fp[key][idx]["proba"])
            depth = float(depths[key])
            b = ast.literal_eval(fp[key][idx]["bbox"])
            b = [int(v) for v in b]
            b = [b[0]/depth,b[1]/wh,b[2]/wh,b[3]/depth,b[4]/wh,b[5]/wh]
            for bv in b:
                assert(bv>=0 and bv <= 1.0)
            vals = vals + b
        for idx in range(len(fp[key]),num_voxels):
            vals = vals + [1.0,-1,-1,-1,-1,-1,-1,-1]
            #vals.append(0.0)
    return vals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='dsb_create_voxel_model_predictions')
    parser.add_argument('--output-folder', help='output folder', required=True);
    parser.add_argument('--labels', help='labels file', default='../stage1_labels.csv')
    parser.add_argument('--stage1-labels', help='stage1 validation labels file', default='../stage1_solution.csv')
    parser.add_argument('--fp', help='voxel false positive predictions', action='append')
    parser.add_argument('--depths', help='image depths', default="../stage1_false_positive/dsb_image_depths.csv");    
    parser.add_argument('--model-file', help='predictions file', default='rf.model.pkl')
    parser.set_defaults(predict=False)
    
    args = parser.parse_args()
    opts = vars(args)

    #load labels
    lmap = {}
    lmap = read_labels(lmap,args.labels)
    lmap = read_labels(lmap,args.stage1_labels)
    depths = read_depths(args.depths);
    
    fps = []
    keys = set()
    #load false positive results
    for filename in args.fp:
        fp = {}
        csvFile = open(filename)
        reader = unicodecsv.DictReader(csvFile,encoding='utf-8')
        for j in reader:
            key = j["key"]
            keys.add(key)
            vid = j["voxel_id"]
            proba = float(j["proba"])
            j["proba"] = proba
            if not key in fp:
                fp[key] = []
            fp[key].append(j)
        for key in fp:
            fp[key] = sorted(fp[key], key=lambda k: k['proba'], reverse=True)
        fps.append(fp)

    #best
    #num_voxels = 2
    num_voxels = 2
    data = []
    target = []
    wh = 512.0
    for key in lmap.keys():
        target.append(lmap[key])
        vals = create_feature_row(key,num_voxels,fps,depths)
        d = np.array(vals)
        d = np.expand_dims(d,axis=0)
        data.append(d)
    X = np.concatenate(data)
    y = np.array(target)
    print X.shape,y.shape

    #a = OptTrain(X, y)
    #a.optimize()
    
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn import svm
    from sklearn.ensemble import GradientBoostingClassifier
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    #clf = LogisticRegression()
    #clf = ExtraTreesClassifier(n_estimators=100)
    clf = RandomForestClassifier(n_estimators=150)
    #clf = GradientBoostingClassifier(n_estimators=40,max_depth=5)
    #clf = svm.SVC(kernel='linear', C=1, probability=True)
    #clf = xgb.XGBClassifier(objective="binary:logistic",nthread=-1,scale_pos_weight=1,n_estimators=50, max_depth=6)

    scores = cross_val_score(clf, X, y, cv=10, scoring="neg_log_loss")
    print scores
    print sum(scores) / float(len(scores))
    scores = cross_val_score(clf, X, y, cv=5, scoring="recall")
    print scores
    print sum(scores) / float(len(scores))
    scores = cross_val_score(clf, X, y, cv=5)
    print scores
    print sum(scores) / float(len(scores))

    data = []
    print "training full model for stage 1"
    model = clf.fit(X,y)
    
    joblib.dump(model,args.output_folder+args.model_file, compress=True)
                
    

