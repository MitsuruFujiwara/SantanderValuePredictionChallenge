import lightgbm as lgb
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, vstack
from sklearn import *
from PIL import Image, ImageDraw, ImageColor

# load dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape, test.shape)

col = [c for c in train.columns if c not in ['ID', 'target']]
xtrain = train[col].copy().values
target = train['target'].values

im = Image.new('RGBA', xtrain.shape)
wh = ImageColor.getrgb('white')
re = ImageColor.getrgb('red')
gr = ImageColor.getrgb('green')
ga = ImageColor.getrgb('gray')

for x in range(xtrain.shape[0]):
    for y in range(xtrain.shape[1]):
        if xtrain[x][y] == 0:
            im.putpixel((x,y), wh)
        elif xtrain[x][y] == target[x]:
            im.putpixel((x,y), re)
        elif (np.abs(xtrain[x][y] - target[x]) / target[x]) < 0.05:
            im.putpixel((x,y), gr)
        else:
            im.putpixel((x,y), ga)
im.save('leak.bmp')

leak_col = []
for c in col:
    leak1 = np.sum((train[c]==train['target']).astype(int))
    leak2 = np.sum((((train[c] - train['target']) / train['target']) < 0.05).astype(int))
    if leak1 > 30 and leak2 > 3500:
        leak_col.append(c)
print(len(leak_col))

col = list(leak_col)
train = train[col +  ['ID', 'target']]
test = test[col +  ['ID']]

#https://www.kaggle.com/johnfarrell/baseline-with-lag-select-fake-rows-dropped
train["nz_mean"] = train[col].apply(lambda x: x[x!=0].mean(), axis=1)
train["nz_max"] = train[col].apply(lambda x: x[x!=0].max(), axis=1)
train["nz_min"] = train[col].apply(lambda x: x[x!=0].min(), axis=1)
train["ez"] = train[col].apply(lambda x: len(x[x==0]), axis=1)
train["mean"] = train[col].apply(lambda x: x.mean(), axis=1)
train["max"] = train[col].apply(lambda x: x.max(), axis=1)
train["min"] = train[col].apply(lambda x: x.min(), axis=1)

test["nz_mean"] = test[col].apply(lambda x: x[x!=0].mean(), axis=1)
test["nz_max"] = test[col].apply(lambda x: x[x!=0].max(), axis=1)
test["nz_min"] = test[col].apply(lambda x: x[x!=0].min(), axis=1)
test["ez"] = test[col].apply(lambda x: len(x[x==0]), axis=1)
test["mean"] = test[col].apply(lambda x: x.mean(), axis=1)
test["max"] = test[col].apply(lambda x: x.max(), axis=1)
test["min"] = test[col].apply(lambda x: x.min(), axis=1)
col += ['nz_mean', 'nz_max', 'nz_min', 'ez', 'mean', 'max', 'min']

for i in range(2, 100):
    train['index'+str(i)] = ((train.index + 2) % i == 0).astype(int)
    test['index'+str(i)] = ((test.index + 2) % i == 0).astype(int)
    col.append('index'+str(i))

train = train.replace(0, np.nan)
test = test.replace(0, np.nan)
train = pd.concat((train, test), axis=0, ignore_index=True)

test['target'] = 0.0
folds = 5
for fold in range(folds):
    x1, x2, y1, y2 = model_selection.train_test_split(train[col], np.log1p(train.target.values), test_size=0.20, random_state=fold)
    params = {'learning_rate': 0.02, 'max_depth': 7, 'boosting': 'gbdt', 'objective': 'regression', 'metric': 'rmse', 'is_training_metric': True, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'seed':fold}
    model = lgb.train(params, lgb.Dataset(x1, label=y1), 3000, lgb.Dataset(x2, label=y2), verbose_eval=200, early_stopping_rounds=100)
    test['target'] += np.expm1(model.predict(test[col], num_iteration=model.best_iteration))
    
test['target'] /= folds
test[['ID', 'target']].to_csv('submission.csv', index=False)
