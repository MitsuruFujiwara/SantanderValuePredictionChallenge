import pandas as pd
import numpy as np

"""
複数モデルのoutputをブレンドして最終的なsubmitファイルを生成するスクリプト。
"""

def main():
    sub = pd.read_csv('sample_submission.csv')
    sub_lgbm = pd.read_csv('submission_lgbm.csv')
    sub_xgb = pd.read_csv('submission_xgb.csv')

    sub['target'] = 0.5*sub_lgbm['target'] + 0.5*sub_xgb['target']

    sub[['ID', 'target']].to_csv('submission_blend.csv', index= False)

if __name__ == '__main__':
    main()
