import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.base import clone


# Load Data
dataset = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical.csv")
dataset = shuffle(dataset, random_state=1)
labels = dataset["deposit"]
dataset.drop(["deposit", "time"], axis=1, inplace=True)


numerical_cols = ["age", "balance", "campaign", "duration"]
categorical_cols = ["marital", "job", "education", "housing", "loan", "month", "day"]
numerical_data = dataset[numerical_cols]
categorical_data = dataset[categorical_cols]

min_max_scaler = MinMaxScaler(feature_range=(0, 1))  # 0-1 scaling
onehot_scaler = OneHotEncoder()

numerical_data = min_max_scaler.fit_transform(X=numerical_data)
categorical_data = onehot_scaler.fit_transform(X=categorical_data).toarray()

print(categorical_data.shape)
print(numerical_data.shape)

dataset = np.concatenate((categorical_data, numerical_data), axis=1)

train_data, test_data, train_label, test_label = train_test_split(dataset, labels, train_size=0.75, random_state=1)

# Models we will use
logistic = linear_model.LogisticRegression(solver='newton-cg', max_iter=1000)
mlp_classifier = MLPClassifier(random_state=1, max_iter=1000, verbose=True)
rbm = BernoulliRBM(random_state=0, verbose=True)

# rbm_features_classifier = Pipeline(
#     steps=[('rbm', rbm), ('logistic', logistic)])
rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('nn', mlp_classifier)]
)

# #############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.1
rbm.n_iter = 50
rbm.batch_size = 128
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 256

mlp_classifier.learning_rate = "adaptive"
mlp_classifier.activation = "logistic"
mlp_classifier.hidden_layer_sizes = (256, 128, 64)

# Training RBM-Logistic Pipeline
rbm_features_classifier.fit(train_data, train_label)

# Training the Logistic regression classifier directly on the pixel
raw_pixel_classifier = clone(mlp_classifier)
# raw_pixel_classifier.C = 100.
raw_pixel_classifier = raw_pixel_classifier.fit(train_data, train_label)
raw_pixel_classifier.fit(train_data, train_label)

# #############################################################################
# Evaluation

Y_pred = rbm_features_classifier.predict(test_data)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(test_label, Y_pred)))

Y_pred = raw_pixel_classifier.predict(test_data)
print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(test_label, Y_pred)))
