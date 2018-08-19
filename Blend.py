import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

"""
複数モデルのoutputをブレンドして最終的なsubmitファイルを生成するスクリプト。
"""

def main():
    # load submission files
    sub = pd.read_csv('sample_submission.csv')
    sub_lgbm = pd.read_csv('submission_lgbm.csv')
    sub_xgb = pd.read_csv('submission_xgb.csv')

    # loada out of fold data
    train_df = pd.read_csv('train.csv')
    oof_lgbm = pd.read_csv('oof_lgbm.csv')
    oof_xgb = pd.read_csv('oof_xgb.csv')

    train_df['lgbm'] = oof_lgbm['OOF_PRED']
    train_df['xgb'] = oof_xgb['OOF_PRED']

    # find best weights
    rmse_bst = 5.0
    for w in np.arange(0,1.001, 0.001):
        _pred = w * train_df['lgbm'] + (1.0-w) * train_df['xgb']
        _rmse = np.sqrt(mean_squared_error(np.log1p(train_df['target']), np.log1p(_pred)))
        if _rmse < rmse_bst:
            rmse_bst = _rmse
            w_bst = (w, 1.0-w)

    print("best w: {}, best rmse: {}".format(w_bst, rmse_bst))

    # take average of each predicted values
    sub['lgbm'] = sub_lgbm['target']
    sub['xgb'] = sub_xgb['target']

    sub['target'] = w_bst[0]*sub_lgbm['target'] + w_bst[1]*sub_xgb['target']

    # save submission file
    sub[['ID', 'target']].to_csv('submission_blend.csv', index= False)

if __name__ == '__main__':
    main()
