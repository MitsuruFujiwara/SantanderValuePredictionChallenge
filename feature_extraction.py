# 参考: https://www.kaggle.com/gurchetan1000/6-new-patterns-extending-jiazhen-0-66-on-train

import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import lightgbm as lgb
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import mode, skew, kurtosis, entropy
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

# adding groups
colgroups = [
    ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'],
    ['266525925', '4b6dfc880', '2cff4bf0c', 'a3382e205', '6488c8200', '547d3135b', 'b46191036', '453128993', '2599a7eb7', '2fc60d4d9', '009319104', 'de14e7687', 'aa31dd768', '2b54cddfd', 'a67d02050', '37aab1168', '939cc02f5', '31f72667c', '6f951302c', '54723be01', '4681de4fd', '8bd53906a', '435f27009', 'f82167572', 'd428161d9', '9015ac21d', 'ec4dc7883', '22c7b00ef', 'd4cc42c3d', '1351bf96e', '1e8801477', 'b7d59d3b5', 'a459b5f7d', '580f5ff06', '39b3c553a', '1eec37deb', '692c44993', 'ce8ce671e', '88ef1d9a8', 'bf042d928'],
    ['9d5c7cb94', '197cb48af', 'ea4887e6b', 'e1d0e11b5', 'ac30af84a', 'ba4ceabc5', 'd4c1de0e2', '6d2ece683', '9c42bff81', 'cf488d633', '0e1f6696a', 'c8fdf5cbf', 'f14b57b8f', '3a62b36bd', 'aeff360c7', '64534cc93', 'e4159c59e', '429687d5a', 'c671db79e', 'd79736965', '2570e2ba9', '415094079', 'ddea5dc65', 'e43343256', '578eda8e0', 'f9847e9fe', '097c7841e', '018ab6a80', '95aea9233', '7121c40ee', '578b81a77', '96b6bd42b', '44cb9b7c4', '6192f193d', 'ba136ae3f', '8479174c2', '64dd02e44', '4ecc3f505', 'acc4a8e68', '994b946ad'],
    ['f1eeb56ae', '62ffce458', '497adaff8', 'ed1d5d137', 'faf7285a1', 'd83da5921', '0231f07ed', '7950f4c11', '051410e3d', '39e1796ab', '2e0148f29', '312832f30', '6f113540d', 'f3ee6ba3c', 'd9fc63fa1', '6a0b386ac', '5747a79a9', '64bf3a12a', 'c110ee2b7', '1bf37b3e2', 'fdd07cac1', '0872fe14d', 'ddef5ad30', '42088cf50', '3519bf4a4', 'a79b1f060', '97cc1b416', 'b2790ef54', '1a7de209c', '2a71f4027', 'f118f693a', '15e8a9331', '0c545307d', '363713112', '73e591019', '21af91e9b', '62a915028', '2ab5a56f5', 'a8ee55662', '316b978cd'],
    ['b26d16167', '930f989bf', 'ca58e6370', 'aebe1ea16', '03c589fd7', '600ea672f', '9509f66b0', '70f4f1129', 'b0095ae64', '1c62e29a7', '32a0342e2', '2fc5bfa65', '09c81e679', '49e68fdb9', '026ca57fd', 'aacffd2f4', '61483a9da', '227ff4085', '29725e10e', '5878b703c', '50a0d7f71', '0d1af7370', '7c1af7bbb', '4bf056f35', '3dd64f4c4', 'b9f75e4aa', '423058dba', '150dc0956', 'adf119b9a', 'a8110109e', '6c4f594e0', 'c44348d76', 'db027dbaf', '1fcba48d0', '8d12d44e1', '8d13d891d', '6ff9b1760', '482715cbd', 'f81c2f1dd', 'dda820122'],
    ['c928b4b74', '8e4d0fe45', '6c0e0801a', '02861e414', 'aac52d8d9', '041c5d0c9', 'd7875bb6c', 'e7c0cfd0f', 'd48c08bda', '0c9462c08', '57dd44c29', 'a93118262', '850027e38', 'db3839ab0', '27461b158', '32174174c', '9306da53f', '95742c2bf', '5831f4c76', '1e6306c7c', '06393096a', '13bdd610a', 'd7d314edc', '9a07d7b1f', '4d2671746', '822e49b95', '3c8a3ced0', '83635fb67', '1857fbccf', 'c4972742d', 'b6c0969a2', 'e78e3031b', '36a9a8479', 'e79e5f72c', '092271eb3', '74d7f2dc3', '277ef93fc', 'b30e932ba', '8f57141ec', '350473311'],
    ['06148867b', '4ec3bfda8', 'a9ca6c2f4', 'bb0408d98', '1010d7174', 'f8a437c00', '74a7b9e4a', 'cfd55f2b6', '632fed345', '518b5da24', '60a5b79e4', '3fa0b1c53', 'e769ee40d', '9f5f58e61', '83e3e2e60', '77fa93749', '3c9db4778', '42ed6824a', '761b8e0ec', 'ee7fb1067', '71f5ab59f', '177993dc6', '07df9f30c', 'b1c5346c4', '9a5cd5171', 'b5df42e10', 'c91a4f722', 'd93058147', '20a325694', 'f5e0f4a16', '5edd220bc', 'c901e7df1', 'b02dfb243', 'bca395b73', '1791b43b0', 'f04f0582d', 'e585cbf20', '03055cc36', 'd7f15a3ad', 'ccd9fc164'],
    ['df838756c', '2cb73ede7', '4dcf81d65', '61c1b7eb6', 'a9f61cf27', '1af4d24fa', 'e13b0c0aa', 'b9ba17eb6', '796c218e8', '37f57824c', 'd1e0f571b', 'f9e3b03b7', 'a3ef69ad5', 'e16a20511', '04b88be38', '99e779ee0', '9f7b782ac', '1dd7bca9f', '2eeadde2b', '6df033973', 'cdfc2b069', '031490e77', '5324862e4', '467bee277', 'a3fb07bfd', '64c6eb1cb', '8618bc1fd', '6b795a2bc', '956d228b9', '949ed0965', 'a4511cb0b', 'b64425521', '2e3c96323', '191e21b5f', 'bee629024', '1977eaf08', '5e645a169', '1d04efde3', '8675bec0b', '8337d1adc'],
    ['a1cd7b681', '9b490abb3', 'b10f15193', '05f54f417', 'a7ac690a8', 'ed6c300c2', 'd0803e3a1', 'b1bb8eac3', 'bd1c19973', 'a34f8d443', '84ec1e3db', '24018f832', '82e01a220', '4c2064b00', '0397f7c9b', 'ba42e41fa', '22d7ad48d', '9abffd22c', 'dbfa2b77f', '2c6c62b54', '9fa38def3', 'ecb354edf', '9c3154ae6', '2f26d70f4', '53102b93f', 'a36b95f78', '1fa0f78d0', '19915a6d3', 'c944a48b5', '482b04cba', '2ce77a58f', '86558e595', 'c3f400e36', '20305585c', 'f8ccfa064', 'dd771cb8e', '9aa27017e', 'cd7f0affd', '236cc1ff5', 'a3fc511cd'],
    ['920a04ee2', '93efdb50f', '15ea45005', '78c57d7cd', '91570fb11', 'c5dacc85b', '145c7b018', '590b24ab1', 'c283d4609', 'e8bd579ae', '7298ca1ef', 'ce53d1a35', 'a8f80f111', '2a9fed806', 'feb40ad9f', 'cfd255ee3', '31015eaab', '303572ae2', 'cd15bb515', 'cb5161856', 'a65b73c87', '71d64e3f7', 'ec5fb550f', '4af2493b6', '18b4fa3f5', '3d655b0ed', '5cc9b6615', '88c0ec0a6', '8722f33bb', '5ed0c24d0', '54f26ee08', '04ecdcbb3', 'ade8a5a19', 'd5efae759', 'ac7a97382', 'e1b20c3a6', 'b0fcfeab8', '438b8b599', '43782ef36', 'df69cf626'],
    ['50603ae3d', '48282f315', '090dfb7e2', '6ccaaf2d7', '1bf2dfd4a', '50b1dd40f', '1604c0735', 'e94c03517', 'f9378f7ef', '65266ad22', 'ac61229b6', 'f5723deba', '1ced7f0b4', 'b9a4f06cd', '8132d18b8', 'df28ac53d', 'ae825156f', '936dc3bc4', '5b233cf72', '95a2e29fc', '882a3da34', '2cb4d123e', '0e1921717', 'c83d6b24d', '90a2428a5', '67e6c62b9', '320931ca8', '900045349', 'bf89fac56', 'da3b0b5bb', 'f06078487', '56896bb36', 'a79522786', '71c2f04c9', '1af96abeb', '4b1a994cc', 'dee843499', '645b47cde', 'a8e15505d', 'cc9c2fc87'],
    ['b6daeae32', '3bdee45be', '3d6d38290', '5a1589f1a', '961b91fe7', '29c059dd2', 'cfc1ce276', '0a953f97e', '30b3daec2', 'fb5f5836e', 'c7525612c', '6fa35fbba', '72d34a148', 'dcc269cfe', 'bdf773176', '469630e5c', '23db7d793', 'dc10234ae', '5ac278422', '6cf7866c1', 'a39758dae', '45f6d00da', '251d1aa17', '84d9d1228', 'b98f3e0d7', '66146c12d', 'd6470c4ce', '3f4a39818', 'f16a196c6', 'b8f892930', '6f88afe65', 'ed8951a75', '371da7669', '4b9540ab3', '230a025ca', 'f8cd9ae02', 'de4e75360', '540cc3cd1', '7623d805a', 'c2dae3a5a'],
    ['d0d340214', '34d3715d5', '9c404d218', 'c624e6627', 'a1b169a3a', 'c144a70b1', 'b36a21d49', 'dfcf7c0fa', 'c63b4a070', '43ebb15de', '1f2a670dd', '3f07a4581', '0b1560062', 'e9f588de5', '65d14abf0', '9ed0e6ddb', '0b790ba3a', '9e89978e3', 'ee6264d2b', 'c86c0565e', '4de164057', '87ba924b1', '4d05e2995', '2c0babb55', 'e9375ad86', '8988e8da5', '8a1b76aaf', '724b993fd', '654dd8a3b', 'f423cf205', '3b54cc2cf', 'e04141e42', 'cacc1edae', '314396b31', '2c339d4f2', '3f8614071', '16d1d6204', '80b6e9a8b', 'a84cbdab5', '1a6d13c4a'],
    ['a9819bda9', 'ea26c7fe6', '3a89d003b', '1029d9146', '759c9e85d', '1f71b76c1', '854e37761', '56cb93fd8', '946d16369', '33e4f9a0e', '5a6a1ec1a', '4c835bd02', 'b3abb64d2', 'fe0dd1a15', 'de63b3487', 'c059f2574', 'e36687647', 'd58172aef', 'd746efbfe', 'ccf6632e6', 'f1c272f04', 'da7f4b066', '3a7771f56', '5807de036', 'b22eb2036', 'b77c707ef', 'e4e9c8cc6', 'ff3b49c1d', '800f38b6b', '9a1d8054b', '0c9b00a91', 'fe28836c3', '1f8415d03', '6a542a40a', 'd53d64307', 'e700276a2', 'bb6f50464', '988518e2d', 'f0eb7b98f', 'd7447b2c5'],
    ['87ffda550', '63c094ba4', '2e103d632', '1c71183bb', 'd5fa73ead', 'e078302ef', 'a6b6bc34a', 'f6eba969e', '0d51722ca', 'ce3d7595b', '6c5c8869c', 'dfd179071', '122c135ed', 'b4cfe861f', 'b7c931383', '44d5b820f', '4bcf15776', '51d4053c7', '1fe5d56b9', 'ea772e115', 'ad009c8b9', '68a945b18', '62fb56487', 'c10f31664', 'cbb673163', 'c8d582dd2', '8781e4b91', 'bd6da0cca', 'ca2b906e8', '11e12dbe8', 'bb0ce54e9', 'c0d2348b7', '77deffdf0', 'f97d9431e', 'a09a238d0', '935ca66a9', '9de83dc23', '861076e21', 'f02ecb19c', '166008929'],
    ['f3cf9341c', 'fa11da6df', 'd47c58fe2', '0d5215715', '555f18bd3', '134ac90df', '716e7d74d', 'c00611668', '1bf8c2597', '1f6b2bafa', '174edf08a', 'f1851d155', '5bc7ab64f', 'a61aa00b0', 'b2e82c050', '26417dec4', '53a550111', '51707c671', 'e8d9394a0', 'cbbc9c431', '6b119d8ce', 'f296082ec', 'be2e15279', '698d05d29', '38e6f8d32', '93ca30057', '7af000ac2', '1fd0a1f2a', '41bc25fef', '0df1d7b9a', '88d29cfaf', '2b2b5187e', 'bf59c51c3', 'cfe749e26', 'ad207f7bb', '11114a47a', '341daa7d1', 'a8dd5cea5', '7b672b310', 'b88e5de84'],
]

def rewrite_compiled_leak(leak_df, lag):
    leak_df["compiled_leak"] = 0
    for i in range(lag):
        c = "leaked_target_"+str(i)
        zeroleak = leak_df["compiled_leak"]==0
        leak_df.loc[zeroleak, "compiled_leak"] = leak_df.loc[zeroleak, c]
    return leak_df

def _get_leak(df, cols, extra_feats, lag=0):
    f1 = cols[:((lag+2) * -1)]
    f2 = cols[(lag+2):]
    for ef in extra_feats:
        f1 += ef[:((lag+2) * -1)]
        f2 += ef[(lag+2):]

    d1 = df[f1].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = df[f2].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})

    d2['pred'] = df[cols[lag]]
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    d4 = d1[~d1.duplicated(['key'], keep=False)]
    d5 = d4.merge(d3, how='inner', on='key')
    d6 = d1.merge(d5, how='left', on='key')

    return d1.merge(d5, how='left', on='key').pred.fillna(0)

def compiled_leak_result(cols, df, transact_cols, l, y):

    max_nlags = len(cols)-2
    df_leak = df[["ID", "target"] + cols]
    df_leak["compiled_leak"] = 0
    df_leak["nonzero_mean"] = df[transact_cols].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
    scores = []
    leaky_value_counts = []
    leaky_value_corrects = []
    leaky_cols = []

    for i in range(max_nlags):
        c = "leaked_target_"+str(i)
        print('Processing lag', i)

        df_leak[c] = _get_leak(df, cols, l, i)
        leaky_cols.append(c)

        df_leak = df_leak.set_index("ID")[leaky_cols+["compiled_leak", "nonzero_mean"]]
        df_leak = df.join(df_leak,on="ID", how="left")
        df_leak = df_leak[["ID", "target"] + cols + leaky_cols+["compiled_leak", "nonzero_mean"]]
        zeroleak = df_leak["compiled_leak"]==0
        df_leak.loc[zeroleak, "compiled_leak"] = df_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(df_leak["compiled_leak"] > 0))
        _correct_counts = sum(df_leak["compiled_leak"]==df_leak["target"])
        leaky_value_corrects.append(_correct_counts*1.0/leaky_value_counts[-1])

        print("Leak values found in train", leaky_value_counts[-1])
        print("% of correct leaks values in train ", leaky_value_corrects[-1])

        tmp = df_leak.copy()
        tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]

        print('Na count',tmp.compiled_leak.isna().sum())

        scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))

        print('Score (filled with nonzero mean)', scores[-1])

    result = dict(score=scores,
                  leaky_count=leaky_value_counts,
                  leaky_correct=leaky_value_corrects,
                  )
    return df_leak, result
    
def getNewDF(num_rows = None):
    # load csv files
    df = pd.read_csv('train_leak.csv', nrows = num_rows, index_col=0)
    test_df = pd.read_csv('test_leak.csv', nrows = num_rows, index_col=0)

    # set test target as nan
    test_df['target'] = np.nan

    # set columns
    feats = [f for f in df.columns if f not in ['ID', 'target']]
    feats_leaked_target = [f for f in df.columns if 'leaked_target_' in f]
    leak_cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f',
                 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5',
                 '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867',
                 'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7',
                 '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
                 '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488',
                 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']

    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))

    # merge test & train
    df = df.append(test_df).reset_index()
    df = df.replace(0, np.nan)

    # add new features
    df['LEAK_COL_MEAN'] = df[leak_cols].mean(axis=1)
    df['LEAK_COL_MAX'] = df[leak_cols].max(axis=1)
    df['LEAK_COL_MIN'] = df[leak_cols].min(axis=1)
    df['LEAK_COL_STD'] = df[leak_cols].std(axis=1)
    df['LEAK_COL_SKEW'] = df[leak_cols].skew(axis=1)
    df['LEAK_COL_KURT'] = df[leak_cols].kurtosis(axis=1)
    df['LEAK_COL_SUM'] = df[leak_cols].sum(axis=1)
    df['LEAK_COL_RANGE'] = df['LEAK_COL_MAX'] - df['LEAK_COL_MIN']
    df['LEAK_COL_NUM_NAN'] = df[leak_cols].isnull().sum(axis=1)
    df['LEAK_COL_MAXMIN_RATIO'] = df['LEAK_COL_MAX'] / df['LEAK_COL_MIN']

    df['TARGET_MEAN'] = df[feats_leaked_target].mean(axis=1)
    df['TARGET_MAX'] = df[feats_leaked_target].max(axis=1)
    df['TARGET_MIN'] = df[feats_leaked_target].min(axis=1)
    df['TARGET_STD'] = df[feats_leaked_target].std(axis=1)
    df['TARGET_SKEW'] = df[feats_leaked_target].skew(axis=1)
    df['TARGET_KURT'] = df[feats_leaked_target].kurtosis(axis=1)
    df['TARGET_SUM'] = df[feats_leaked_target].sum(axis=1)
    df['TARGET_RANGE'] = df['TARGET_MAX'] - df['TARGET_MIN']
    df['TARGET_NUM_NAN'] = df[feats_leaked_target].isnull().sum(axis=1)
    df['TARGET_MAXMIN_RATIO'] = df['TARGET_MAX'] / df['TARGET_MIN']

    for i in range(2,100):
        df['index'+str(i)] = ((df['index'] + 2) % i == 0).astype(int)

    del test_df

    return df

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    transact_cols = [f for f in train.columns if f not in ["ID", "target"]]
    trY = np.log1p(train["target"]).values
    test["target"] = train["target"].mean()
    valY = np.log1p(test["target"]).values

    all_df = pd.concat([train, test]).reset_index(drop=True)
    all_df.columns = all_df.columns.astype(str)

    cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f',
            'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5',
            '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867',
            'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7',
            '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
            '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488',
            'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']

    pattern_1964666 = pd.read_csv('pattern_1964666.66.csv')
    pattern_1166666 = pd.read_csv('pattern_1166666.66.csv')
    pattern_812666 = pd.read_csv('pattern_812666.66.csv')
    pattern_2002166 = pd.read_csv('pattern_2002166.66.csv')
    pattern_3160000 = pd.read_csv('pattern_3160000.csv')
    pattern_3255483 = pd.read_csv('pattern_3255483.88.csv')

    pattern_1964666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
    pattern_1166666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
    pattern_812666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
    pattern_2002166.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
    pattern_3160000.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
    pattern_3255483.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)

    pattern_1166666.rename(columns={'8.50E+43': '850027e38'},inplace=True)

    l=[]
    l.append(pattern_1964666.columns.values.tolist())
    l.append(pattern_1166666.columns.values.tolist())
    l.append(pattern_812666.columns.values.tolist())
    l.append(pattern_2002166.columns.values.tolist())
    l.append(pattern_3160000.columns.values.tolist())
    l.append(pattern_3255483.columns.values.tolist())

    # add col groups
    l = l + colgroups

    train_leak, train_result = compiled_leak_result(cols, train, transact_cols, l, trY)
    leaky_cols = [c for c in train_leak.columns if 'leaked_target_' in c]
    train_result = pd.DataFrame.from_dict(train_result, orient='columns')

    train_result.to_csv('train_leaky_stat.csv', index=False)
    train_leak.to_csv('train_leak.csv')

    best_score = np.min(train_result['score'])
    best_lag = np.argmin(train_result['score'])
    print('best_score', best_score, '\nbest_lag', best_lag)

    test_leak, test_result = compiled_leak_result(cols, test, transact_cols, l, valY)
    test_result = pd.DataFrame.from_dict(test_result, orient='columns')

    # get submission data
    test_leak = rewrite_compiled_leak(test_leak, best_lag)
    test_leak.loc[test_leak["compiled_leak"]==0, "compiled_leak"] = test_leak.loc[test_leak["compiled_leak"]==0, "nonzero_mean"]

    test_result.to_csv('test_leaky_stat.csv', index=False)
    test_leak.to_csv('test_leak.csv')

    test_res = test_leak[leaky_cols+['compiled_leak']].replace(0.0, np.nan)

    # submission
    sub = test[["ID"]]
    sub["target"] = test_leak["compiled_leak"]
    sub.to_csv("submission_baseline.csv", index=False)

    print(sub)

if __name__ == '__main__':
    main()
