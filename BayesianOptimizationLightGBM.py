import gc
import pandas as pd
import numpy as np
import lightgbm

from bayes_opt import BayesianOptimization
from feature_extraction import getNewDF

# 以下参考
# https://github.com/fmfn/BayesianOptimization
# https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code

NUM_ROWS=None

# load dataset
DF = getNewDF(NUM_ROWS)

# split test & train
TRAIN_DF = DF[DF['target'].notnull()]
FEATS = [f for f in DF.columns if f not in ['ID', 'target']]

lgbm_train = lightgbm.Dataset(TRAIN_DF[FEATS],
                              np.log1p(TRAIN_DF['target']),
                              free_raw_data=False
                              )

del DF, TRAIN_DF

def lgbm_eval(num_leaves,
              colsample_bytree,
              subsample,
              max_depth,
              reg_alpha,
              reg_lambda,
              min_split_gain,
              min_child_weight,
              min_data_in_leaf
              ):

    params = dict()
    params['task'] = 'train',
    params['boosting'] = 'gbdt',
    params['objective'] = 'regression',
    params['metric'] = 'rmse',
    params["learning_rate"] = 0.02
    params['device'] = 'gpu'
    params['seed']=326,
    params['bagging_seed']=326

    params["num_leaves"] = int(num_leaves)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['subsample'] = max(min(subsample, 1), 0)
    params['max_depth'] = int(max_depth)
    params['reg_alpha'] = max(reg_alpha, 0)
    params['reg_lambda'] = max(reg_lambda, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    params['min_data_in_leaf'] = int(min_data_in_leaf)
    params['verbose']=-1

    clf = lightgbm.cv(params=params,
                      train_set=lgbm_train,
                      metrics=["rmse"],
                      nfold=5,
                      folds=None,
                      num_boost_round=10000, # early stopありなのでここは大きめの数字にしてます
                      early_stopping_rounds=200,
                      verbose_eval=100,
                      seed=47,
                     )
    gc.collect()
    return -clf['rmse-mean'][-1]

def main():

    # clf for bayesian optimization
    clf_bo = BayesianOptimization(lgbm_eval, {'num_leaves': (32, 100),
                                              'colsample_bytree': (0.001, 1),
                                              'subsample': (0.001, 1),
                                              'max_depth': (7, 15),
                                              'reg_alpha': (8, 20),
                                              'reg_lambda': (0, 10),
                                              'min_split_gain': (0, 1),
                                              'min_child_weight': (0, 45),
                                              'min_data_in_leaf': (0, 100),
                                              })

    clf_bo.maximize(init_points=15, n_iter=25)

    res = pd.DataFrame(clf_bo.res['max']['max_params'], index=['max_params'])

    res.to_csv('max_params_lgbm.csv')

if __name__ == '__main__':
    main()
