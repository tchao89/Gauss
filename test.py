# coding: utf-8
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error

import core.lightgbm as lgb

print('Loading data...')
# load or create your dataset
regression_example_dir = "/home/liangqian/Gauss/"
df_train = pd.read_csv(str(regression_example_dir + 'regression.train'), header=None, sep='\t')
df_test = pd.read_csv(str(regression_example_dir + 'regression.test'), header=None, sep='\t')

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')
init_model = lgb.Booster(train_set=lgb_train, model_file="/home/liangqian/Gauss/model.txt")
# train
gbm = lgb.train(params,
                lgb_train,
                init_model=init_model,
                keep_training_booster=True,
                num_boost_round=20)

print('Saving model...')
# save model to file
gbm.save_model('model_2.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
print(f'The RMSE of prediction is: {rmse_test}')
