import gc
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from contextlib import contextmanager
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Input
from keras.layers.core import Dropout
from keras.regularizers import l2
from keras.optimizers import SGD

from feature_extraction import getNewDF

"""
LightGBMと同じ特徴量でDNNの予測値を出すためのスクリプト。
最終段階でアンサンブルモデルの一部として使うかもしれないので作っておきます。
"""

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def autoencoder(encoding_dim, decoding_dim, activation, X, nb_epoch):
    # set parameters
    input_data = Input(shape=(encoding_dim,))

    # set layer
    encoded = Dense(decoding_dim, activation=activation, W_regularizer=l2(0.0001))(input_data)
    decoded = Dense(encoding_dim, activation=activation, W_regularizer=l2(0.0001))(encoded)

    # set autoencoder
    _autoencoder = Model(input=input_data, output=decoded)
    _encoder = Model(input=input_data, output=encoded)

    # compile
    _autoencoder.compile(loss='mse', optimizer='adam')

    # fit autoencoder
    _autoencoder.fit(X,X, nb_epoch=nb_epoch, verbose=1)

    return _encoder

def _model(input_dim):
    """
    モデルの定義
    モデルのパラメータなど変える場合は基本的にこの中をいじればおｋ
    """
    model = Sequential()
    model.add(Dense(output_dim=1000, input_dim=input_dim, W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(output_dim=1000, input_dim=1000, W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1000, input_dim=1000, W_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1, input_dim=1000, W_regularizer=l2(0.0001)))
    model.add(Activation('linear'))
    return model

def getInitialWeights(trX, num_epoc, use_saved_params=False):

    print("Starting AutoEncoder. Train shape: {}".format(trX[0].shape))

    # 各層のinitial weightsを取得するため空のモデルを生成しておきます
    base_model = _model(trX[0].shape[1])

    # モデル各層の入力・出力値を取得し、事前学習用のdimsを定義
    w = base_model.get_weights()
    dims = [w[0].shape[0]] + [_w.shape[0] for _w in w[1::2]]

    if use_saved_params:
        # 保存済みのパラメータを使う場合
        for i, t in enumerate(dims[:-1]):
            _encoder = load_model('encoder' + str(i) + '.h5')
            w[i*2] = _encoder.get_weights()[0]
            w[i*2+1] = _encoder.get_weights()[1]

            del _encoder
    else:
        # 新たにAuto Encoderにより各層のweightを求める場合
        encoders = []
        for i, t in enumerate(dims[:-1]):
            _X = trX[i]
            # fit autoencoder
            _encoder = autoencoder(t, dims[i+1], 'relu', _X, num_epoc)

            # save fitted encoder
            encoders.append(_encoder)
            _encoder.save('encoder' + str(i) + '.h5')

            # generate predicted value (for next encoder)
            trX.append(_encoder.predict(_X))

            del _encoder, _X

        # set initial weights
        for i, e in enumerate(encoders):
            w[i*2] = e.get_weights()[0]
            w[i*2+1] = e.get_weights()[1]

        del encoders
    del base_model
    return w

def kfold_dnn(df, num_folds, stratified = False, debug= False, use_saved_params=False):
    """
    DNN用の前処理など
    """
    # set feature columns
    feats = [f for f in df.columns if f not in ['ID', 'target']]

    # 欠損値を平均で埋めておきます
    df[feats] = df[feats].astype('float64')
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df[feats] = df[feats].fillna(df[feats].mean())

    # DNN用のスケーリング
    ms = MinMaxScaler()
    df_ms = pd.DataFrame(ms.fit_transform(df[feats]), columns=feats, index=df.index)
    df_ms[['ID', 'target']]=df[['ID', 'target']]

    # 事前学習でモデルの初期値を求める #ここではTESTデータも含めて全てのデータを使います。
    trX = [np.array(df_ms[feats])]
    weights = getInitialWeights(trX, num_epoc=5, use_saved_params=use_saved_params)

    """
    k-foldによるDNNモデルの推定
    """
    # Divide in training/validation and test data
    train_df = df_ms[df_ms['target'].notnull()]
    test_df = df_ms[df_ms['target'].isnull()]

    del df, df_ms
    gc.collect()

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)

    print("Starting DNN. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    # K-folds
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], np.log1p(train_df['target'].iloc[train_idx])
        valid_x, valid_y = train_df[feats].iloc[valid_idx], np.log1p(train_df['target'].iloc[valid_idx])

        # set model
        model = _model(train_x.shape[1])

        # set early stopping
        es_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

        # set model weights
        model.set_weights(weights)

        # compile model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

        # training
        history = model.fit(train_x, train_y, nb_epoch=1000, verbose=2,
                            validation_data=(valid_x, valid_y),
                            callbacks=[es_cb])

        oof_preds[valid_idx] = np.expm1(model.predict(valid_x).reshape(valid_y.shape[0]))
        sub_preds += np.expm1(model.predict_proba(test_df[feats]).reshape(test_df.shape[0])) / folds.n_splits

        print('Fold %2d RMSE : %.6f' % (n_fold + 1, np.sqrt(mean_squared_error(valid_y, np.log1p(oof_preds[valid_idx])))))
        del model, train_x, train_y, valid_x, valid_y, history
        gc.collect()

    print('Full RMSE score %.6f' % np.sqrt(mean_squared_error(np.log1p(train_df['target']), np.log1p(oof_preds))))

    if not debug:
        # 提出データの予測値を保存
        test_df['target'] = sub_preds
        test_df[['ID', 'target']].to_csv(submission_file_name, index= False)

        # out of foldの予測値を保存
        train_df['OOF_PRED'] = oof_preds
        train_df[['ID', 'OOF_PRED']].to_csv(oof_file_name, index= False)

def main(debug = False, use_saved_params=False):
    num_rows = 10000 if debug else None
    with timer("Process generating new data"):
        df = getNewDF(num_rows)
        gc.collect()
    with timer("Run DNN with kfold"):
        kfold_dnn(df, num_folds= 10, stratified=False, debug= debug, use_saved_params=use_saved_params)

if __name__ == '__main__':
    submission_file_name = "submission_dnn.csv"
    oof_file_name = "oof_dnn.csv"
    main(use_saved_params=True)
