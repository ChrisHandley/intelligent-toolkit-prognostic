import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

# toolkit
from toolkit import create_benchmark


# create the inputs and outputs
X, y = make_regression(n_samples=1000, n_features=20, noise=0.2, n_targets = 1)

print(X.shape)
print(y.shape)
# reshape y
y = np.reshape(y, (-1,1))
print(y.shape)
#exit()
print("Load data")

z = pd.read_csv('Results_06042020.csv')

data_outputs1 = z.iloc[:,0].values
data_outputs2 = z.iloc[:,2].values

data_outputs1 = np.reshape(data_outputs1, (-1,1))
data_outputs2 = np.reshape(data_outputs2, (-1,1))

data_outputs = np.concatenate((data_outputs1, data_outputs2), axis=1)
#print(data_outputs.shape)
#print(data_outputs)
#exit()
data_inputs = z.iloc[:,3:7].values

X = data_inputs[:,:]
y = data_outputs[:,:]
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)




#def base_model():
#    model = tf.keras.Sequential([
#    layers.Dense(64, activation='tanh', input_shape=(X_train.shape[1],),bias_regularizer=tf.keras.regularizers.l2(0.01)),
#    layers.Dense(64, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
#    layers.Dense(32, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
#    layers.Dense(16, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
#    layers.Dense(8, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
#    layers.Dense(6, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
#    layers.Dense(4, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
#    layers.Dense(1, activation='linear')])
#
#    radam = tfa.optimizers.RectifiedAdam()
#    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
#    model.compile(optimizer=ranger,
#                  loss='mse',
#                  metrics=['mae'])

#    return model


def base_model():
    model = tf.keras.Sequential([
        layers.Dense(40, activation='tanh', input_shape=(X_train.shape[1],),bias_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(40, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(40, activation='tanh',bias_regularizer=tf.keras.regularizers.l2(0.01)),
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
                          epochs=1000,
                          batch_size=32,
                          verbose=0,
                          validation_split = 0.1)


print(X.shape,y.shape)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


custom_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

for i in range(data_outputs.shape[1]):
    m_train = y_train[:,i]
    m_test = y_test[:,i]
    m = y[:,i]
    # reshape y_train and y_test for tensorflow
    m_train = np.reshape(m_train, (-1,1))
    m_test = np.reshape(m_test, (-1,1))
    m = np.reshape(m, (-1,1))
    print(m_train.shape)
    print(m_test.shape)
    print(m.shape)

    print('Train', X_train.shape, m_train.shape, 'Test', X_test.shape, m_test.shape)
    create_benchmark(X, m, new_model_list=[dnn_model], new_model_name_list=['DNN'], custom_scorer_list=[custom_scorer],
                 custom_scorer_name_list=['mape'], num_cv_fold=2)
    print("next phase")
    exit()
exit()







