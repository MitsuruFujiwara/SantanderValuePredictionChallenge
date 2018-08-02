import lightgbm as lgb
import pandas as pd
import numpy as np

from sklearn import *
from PIL import Image, ImageDraw, ImageColor

# load dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape, test.shape)

# set columns
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
