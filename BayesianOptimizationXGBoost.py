import gc
import numpy as np
import pandas as pd
import xgboost

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

xgb_train = xgboost.DMatrix(TRAIN_DF[FEATS],
                        np.log1p(TRAIN_DF['target'])
                        )

del DF, TRAIN_DF

def xgb_eval(gamma,
             max_depth,
             min_child_weight,
             subsample,
             colsample_bytree,
             colsample_bylevel,
             alpha,
             _lambda):

    params = {
            'objective':'gpu:reg:linear', # GPU parameter
            'booster': 'gbtree',
            'eval_metric':'rmse',
            'silent':1,
            'eta': 0.02,
            'tree_method': 'gpu_hist', # GPU parameter
            'predictor': 'gpu_predictor', # GPU parameter
            'seed':326
            }

    params['gamma'] = gamma
    params['max_depth'] = int(max_depth)
    params['min_child_weight'] = min_child_weight
    params['subsample'] = max(min(subsample, 1), 0)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['colsample_bylevel'] = max(min(colsample_bylevel, 1), 0)
    params['alpha'] = max(alpha, 0)
    params['lambda'] = max(_lambda, 0)

    clf = xgboost.cv(params=params,
                     dtrain=xgb_train,
                     num_boost_round=10000, # early stopありなのでここは大きめの数字にしてます
                     nfold=5,
                     metrics=["rmse"],
                     folds=None,
                     early_stopping_rounds=200,
                     verbose_eval=100,
                     seed=47,
                     )
    gc.collect()
    return -clf['test-rmse-mean'].iloc[-1]

def main():
    # clf for bayesian optimization
    clf_bo = BayesianOptimization(xgb_eval, {'gamma':(0, 1),
                                             'max_depth': (6, 6),
                                             'min_child_weight': (0, 45),
                                             'subsample': (0.001, 1),
                                             'colsample_bytree': (0.001, 1),
                                             'colsample_bylevel': (0.001, 1),
                                             'alpha': (9, 20),
                                             '_lambda': (0, 10)
                                             })

    clf_bo.maximize(init_points=15, n_iter=25)

    res = pd.DataFrame(clf_bo.res['max']['max_params'], index=['max_params'])

    res.to_csv('max_params_xgb.csv')

if __name__ == '__main__':
    main()
