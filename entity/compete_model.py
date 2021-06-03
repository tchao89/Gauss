import nni
import logging
from catboost import CatBoost
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
import lightgbm as lgb
from sklearn.metrics import f1_score

LOG = logging.getLogger('auto-gbdt')


class tree_model:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.dataset = [x_train, x_test, y_train, y_test]
        self.hpo_step = 100

    def run(self):
        pass

    def xgboost_model(self):
        pass

    def lightgbm_model(self):
        best_metrics = -1
        best_params = None
        best_model = None

        def get_default_parameters():
            lightgbm_params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'num_class': 1,
                'metric': 'auc',
                'num_leaves': 32,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 1,
                'max_depth': 9,
                'nthread': -1,
            }
            return lightgbm_params

        def load_data(x_train, x_test, y_train, y_test):
            x_eval = x_test
            y_eval = y_test
            # create dataset for lightgbm
            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_eval = lgb.Dataset(x_eval, y_eval, reference=lgb_train)

            return lgb_train, lgb_eval, x_test, y_test

        def run(params, lgb_train, lgb_eval, X_test, y_test):
            params['num_leaves'] = int(params['num_leaves'])

            boost_model = lgb.train(params,
                                    lgb_train,
                                    num_boost_round=1000,
                                    valid_sets=lgb_eval,
                                    early_stopping_rounds=5, verbose_eval=1)

            y_pred = boost_model.predict(X_test, num_iteration=boost_model.best_iteration)
            pred_class = [1 if item > 0.5 else 0 for item in y_pred]
            eval_metrics = f1_score(y_true=y_test, y_pred=pred_class)
            return boost_model, eval_metrics

        train_set, test_set, train_data, test_data = load_data(self.dataset[0], self.dataset[1], self.dataset[2],
                                                               self.dataset[3])

        try:
            tuner = HyperoptTuner("tpe")
            tuner.update_search_space({
                "num_leaves": {"_type": "randint",
                               "_value": [10, 20, 30, 40, 50, 60]
                               },
                "learning_rate": {"_type": "choice",
                                  "_value": [0.01, 0.05, 0.1, 0.2, 0.3]
                                  },
                "bagging_fraction": {"_type": "uniform",
                                     "_value": [0.7, 1.0]
                                     },
                "feature_fraction": {"_type": "uniform",
                                     "_value": [0.7, 1.0]
                                     },
                "bagging_freq": {"_type": "choice",
                                 "_value": [1, 2, 4, 8, 10]
                                 },
                "max_depth": {"_type": "choice",
                              "_value": [3, 5, 6, 7, 8, 9, 12]
                              },
                "lambda_l2": {
                    "_type": "uniform",
                    "_value": [0, 0.9]
                }
            })

            for k in range(self.hpo_step):
                PARAMS = get_default_parameters()
                RECEIVED_PARAMS = tuner.generate_parameters(k)
                PARAMS.update(RECEIVED_PARAMS)
                model, new_metrics = run(PARAMS, train_set, test_set, train_data, test_data)
                print('new metrics is: ', new_metrics)
                tuner.receive_trial_result(k, RECEIVED_PARAMS, new_metrics)
                if new_metrics > best_metrics:
                    best_model = model
                    best_metrics = new_metrics
                    best_params = PARAMS

        except Exception as exception:
            LOG.exception(exception)
            raise
        print('best metrics is: ', best_metrics)
        best_model.save_model("/home/liangqian/PycharmProjects/TreeModelIntegrated/compete.txt")
        return best_model, best_params

    def catboost_model(self, cat_list=None):
        best_metrics = 0
        best_model = None

        def get_default_parameters():
            catboost_parameters = {'thread_count': -1,
                                   'learning_rate': 0.01,
                                   'eval_metric': 'AUC',
                                   'max_depth': 12,
                                   'loss_function': 'Logloss',
                                   'verbose': False,
                                   'iterations': 600,
                                   'subsample': 0.8,
                                   'colsample_bylevel': 0.8,
                                   'early_stopping_rounds': 5
                                   }
            return catboost_parameters

        def load_data():
            return self.dataset

        def run(params: dict, cat_features: list, dataset: list):
            boost_model = CatBoost(params=params)
            boost_model.fit(dataset[0], dataset[2], cat_features=cat_features, early_stopping_rounds=3,
                            eval_set=(dataset[1], dataset[3]))
            pred_class = boost_model.predict(dataset[1], prediction_type='Class')
            eval_metrics = f1_score(y_true=dataset[3], y_pred=pred_class, average="micro")

            nni.report_final_result(eval_metrics)
            return boost_model, eval_metrics

        cat_dataset = load_data()

        try:
            tuner = HyperoptTuner("anneal")
            tuner.update_search_space({
                "learning_rate": {
                    "_type": "choice",
                    "_value": [0.01, 0.05, 0.1, 0.2, 0.3]
                },
                "iterations": {
                    "_type": "choice",
                    "_value": [100, 300, 500, 700, 1000]
                },
                "subsample":
                    {
                        "_type": "uniform",
                        "_value": [0.7, 1.0]
                    },
                "colsample_bylevel":
                    {
                        "_type": "uniform",
                        "_value": [0.7, 1.0]
                    },
                "max_depth":
                    {
                        "_type": "choice",
                        "_value": [3, 5, 6, 7, 8, 9, 12]
                    }
            })

            for k in range(self.hpo_step):
                PARAMS = get_default_parameters()
                RECEIVED_PARAMS = tuner.generate_parameters(k)
                PARAMS.update(RECEIVED_PARAMS)
                model, new_metrics = run(PARAMS, cat_features=cat_list, dataset=cat_dataset)
                tuner.receive_trial_result(k, RECEIVED_PARAMS, new_metrics)

                if new_metrics > best_metrics:
                    best_model = model
                    best_metrics = new_metrics

        except Exception as exception:
            LOG.exception(exception)
            raise
        print('best metrics is: ', best_metrics)
        return best_model

    def xgboost_lr_model(self):
        pass

    def lightgbm_lr_model(self):
        pass

    def catboost_lr_model(self):
        pass

    def xgboost_fm_model(self):
        pass

    def xdeepfm_model(self):
        pass
