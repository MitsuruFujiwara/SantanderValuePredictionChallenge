# Forked from kernel : https://www.kaggle.com/the1owl/love-is-the-answer

import lightgbm as lgb
import pandas as pd
import numpy as np
import gc
import time

from contextlib import contextmanager
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def getLeakCol(data):
    cols = [c for c in data.columns if c not in ['ID', 'target', 'index']]
    for c in cols:
        leak1 = np.sum((data[c]==data['target']).astype(int))
        leak2 = np.sum((((data[c] - data['target']) / data['target']) < 0.05).astype(int))
        if leak1 > 30 and leak2 > 3500:
            yield c

def display_importances(feature_importance_df_, outputpath, csv_outputpath):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    # importance下位の確認用に追加しました
    _feature_importance_df_=feature_importance_df_.groupby('feature').sum()
    _feature_importance_df_.to_csv(csv_outputpath)

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

def getNewDF(num_rows = None):
    # load csv files
    train = pd.read_csv('train.csv', nrows = num_rows)
    test = pd.read_csv('test.csv', nrows = num_rows)
    print("Train samples: {}, test samples: {}".format(len(train), len(test)))

    # get leak cols
    leak_col = list(getLeakCol(train))
    print("Number of Leak Cols:", len(leak_col))

    # get new train & test data
    train = train[leak_col +  ['ID', 'target']]
    test = test[leak_col +  ['ID']]

    # add new features1 # https://www.kaggle.com/johnfarrell/baseline-with-lag-select-fake-rows-dropped
    train["nz_mean"] = train[leak_col].apply(lambda x: x[x!=0].mean(), axis=1)
    train["nz_max"] = train[leak_col].apply(lambda x: x[x!=0].max(), axis=1)
    train["nz_min"] = train[leak_col].apply(lambda x: x[x!=0].min(), axis=1)
    train["ez"] = train[leak_col].apply(lambda x: len(x[x==0]), axis=1)
    train["mean"] = train[leak_col].apply(lambda x: x.mean(), axis=1)
    train["max"] = train[leak_col].apply(lambda x: x.max(), axis=1)
    train["min"] = train[leak_col].apply(lambda x: x.min(), axis=1)

    test["nz_mean"] = test[leak_col].apply(lambda x: x[x!=0].mean(), axis=1)
    test["nz_max"] = test[leak_col].apply(lambda x: x[x!=0].max(), axis=1)
    test["nz_min"] = test[leak_col].apply(lambda x: x[x!=0].min(), axis=1)
    test["ez"] = test[leak_col].apply(lambda x: len(x[x==0]), axis=1)
    test["mean"] = test[leak_col].apply(lambda x: x.mean(), axis=1)
    test["max"] = test[leak_col].apply(lambda x: x.max(), axis=1)
    test["min"] = test[leak_col].apply(lambda x: x.min(), axis=1)

    # add new features2
    for i in range(2, 100):
        train['index'+str(i)] = ((train.index + 2) % i == 0).astype(int)
        test['index'+str(i)] = ((test.index + 2) % i == 0).astype(int)

    # concat train & test
    df = pd.concat((train, test), axis=0, ignore_index=True)
    del train, test

    # replace zero value as nan
    df = df.replace(0, np.nan)

    return df

def kfold_lightgbm(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['ID', 'target', 'index']]

    # k-fold cross validation
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], np.log1p(train_df['target'].iloc[train_idx])
        valid_x, valid_y = train_df[feats].iloc[valid_idx], np.log1p(train_df['target'].iloc[valid_idx])

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        params ={
                'device' : 'gpu',
                'gpu_use_dp':True, #これで倍精度演算できるっぽいです
                'task': 'train',
                'learning_rate': 0.02,
                'max_depth': 7,
                'boosting': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'num_iteration': 10000,
                'is_training_metric': True,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed':326,
                'bagging_seed':326,
                'drop_seed':326
                }

        clf = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds[valid_idx] = np.expm1(clf.predict(valid_x, num_iteration=clf.best_iteration))
        sub_preds += np.expm1(clf.predict(test_df[feats], num_iteration=clf.best_iteration)) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain', iteration=clf.best_iteration)
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, np.sqrt(mean_squared_error(np.expm1(valid_y), oof_preds[valid_idx]))))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full RMSE score %.6f' % np.sqrt(mean_squared_error(train_df['target'], oof_preds)))

    if not debug:
        # output final prediction
        test_df['target'] = sub_preds
        test_df[['ID', 'target']].to_csv(submission_file_name, index= False)

    return feature_importance_df

def main(debug = False):
    num_rows = 10000 if debug else None
    with timer("Process generating new data"):
        df = getNewDF(num_rows)
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified=False, debug= debug)
        display_importances(feat_importance ,'lgbm_importances.png', 'feature_importance.csv')

if __name__ == '__main__':
    submission_file_name = "submission.csv"
    main()
