import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# toolkit
from toolkit import create_benchmark

print("CITRUS TEST Begins")
# create the inputs and outputs
#X, y = make_regression(n_samples=1000, n_features=20, noise=0.2, n_targets = 1)
X, y = make_regression(n_samples=1000, n_features=20, noise=0.2, n_targets = 1)
#print(X)
#print(X.shape) inputs
#print(y.shape)
#print(X.shape)
z = pd.read_csv('Results_06042020.csv')
#print(z)
##
## Split into inputs and outputs
##

outputs = z.iloc[:,0:3].values
inputs = z.iloc[:,3:7].values
print(inputs.shape)

# reshape y
#y = np.reshape(y, (-1,1))
#print(y.shape)
#exit()

# split
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.20)
#print(inputs.shape)
#print(X_train.shape)
#exit()

# reshape y_train and y_test for tensorflow
#y_train = np.reshape(y_train, (-1,1))
#y_test = np.reshape(y_test, (-1,1))
#print(y_train.shape)
#exit()
print('Original', z.shape, inputs.shape, outputs.shape)
print('Train', X_train.shape, y_train.shape, 'Test', X_test.shape, y_test.shape)

def base_model():
    model = tf.keras.Sequential([
    layers.Dense(64, activation='tanh', input_shape=(X_train.shape[1],),bias_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(64, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(32, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(16, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(8, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(6, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(4, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(1, activation='linear')])

    radam = tfa.optimizers.RectifiedAdam()
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    model.compile(optimizer=ranger,
                  loss='mse',
                  metrics=['mae'])

    return model

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True)

dnn_model = KerasRegressor(build_fn=base_model,
                          epochs=200,
                          batch_size=32,
                          verbose=0,
                          validation_split = 0.1)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

custom_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

create_benchmark(X, y, new_model_list=[dnn_model], new_model_name_list=['DNN'], custom_scorer_list=[custom_scorer],
                 custom_scorer_name_list=['mape'], num_cv_fold=2)

