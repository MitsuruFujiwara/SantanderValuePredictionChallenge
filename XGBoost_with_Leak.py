# Forked from kernel : https://www.kaggle.com/the1owl/love-is-the-answer

import pandas as pd
import numpy as np
import gc
import time
import xgboost as xgb

from contextlib import contextmanager
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from feature_extraction import getNewDF

warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def display_importances(feature_importance_df_, outputpath, csv_outputpath):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    # importance下位の確認用に追加しました
    _feature_importance_df_=feature_importance_df_.groupby('feature').sum()
    _feature_importance_df_.to_csv(csv_outputpath)

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('XGBoost Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

def kfold_xgboost(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    print("Starting XGBoost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
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
    feats = [f for f in train_df.columns if f not in ['ID', 'target']]

    # final predict用にdmatrix形式のtest dfを作っておきます
    test_df_dmtrx = xgb.DMatrix(test_df[feats], label=train_df['target'])

    # k-fold cross validation
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], np.log1p(train_df['target'].iloc[train_idx])
        valid_x, valid_y = train_df[feats].iloc[valid_idx], np.log1p(train_df['target'].iloc[valid_idx])

        # set data structure
        xgb_train = xgb.DMatrix(train_x,
                                label=train_y)
        xgb_test = xgb.DMatrix(valid_x,
                               label=valid_y)

        params = {
                'objective':'gpu:reg:linear', # GPU parameter
                'booster': 'gbtree',
                'eval_metric':'rmse',
                'silent':1,
                'eta': 0.02,
                'max_depth': 6,
                'min_child_weight': 42.2260138575,
                'gamma': 0.5997057606,
                'subsample': 0.6221326906,
                'colsample_bytree': 0.6405879054,
                'colsample_bylevel': 0.9772125093,
                'alpha':16.3131297094,
                'lambda': 2.9672501863,
                'tree_method': 'gpu_hist', # GPU parameter
                'predictor': 'gpu_predictor', # GPU parameter
                'seed':int(2**n_fold)
                }

        clf = xgb.train(
                        params,
                        xgb_train,
                        num_boost_round=10000,
                        evals=[(xgb_train,'train'),(xgb_test,'test')],
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds[valid_idx] = np.expm1(clf.predict(xgb_test))
        sub_preds += np.expm1(clf.predict(test_df_dmtrx)) / folds.n_splits

        fold_importance_df = pd.DataFrame.from_dict(clf.get_score(importance_type='gain'), orient='index', columns=['importance'])
        fold_importance_df["feature"] = fold_importance_df.index.tolist()
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, np.sqrt(mean_squared_error(valid_y, np.log1p(oof_preds[valid_idx])))))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full RMSE score %.6f' % np.sqrt(mean_squared_error(np.log1p(train_df['target']), np.log1p(oof_preds))))

    if not debug:
        # 提出データの予測値を保存
        test_df['target'] = sub_preds
        test_df[['ID', 'target']].to_csv(submission_file_name, index= False)

        # out of foldの予測値を保存
        train_df['OOF_PRED'] = oof_preds
        train_df[['ID', 'OOF_PRED']].to_csv(oof_file_name, index= False)

    return feature_importance_df

def main(debug = False):
    num_rows = 10000 if debug else None
    with timer("Process generating new data"):
        df = getNewDF(num_rows)
        gc.collect()
    with timer("Run XGBoost with kfold"):
        feat_importance = kfold_xgboost(df, num_folds= 10, stratified=False, debug= debug)
        display_importances(feat_importance ,'xgb_importances.png', 'feature_importance_xgb.csv')

if __name__ == '__main__':
    submission_file_name = "submission_xgb.csv"
    oof_file_name = "oof_xgb.csv"
    main()
