import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

# toolkit
from toolkit import create_benchmark

print("CITRUS TEST Begins")
# create the inputs and outputs
z = pd.read_csv('Results_06042020.csv')
#print(z)
##
## Split into inputs and outputs
##

data_outputs = z.iloc[:,0:3].values
data_inputs = z.iloc[:,3:7].values
print(data_inputs.shape)

model = tf.keras.Sequential()
model.add(layers.Dense(30, activation='tanh', input_shape=(data_inputs.shape[1],), name="layer1"))
model.add(layers.Dense(30, activation="tanh", name="layer2"))
model.add(layers.Dense(3, name="layer3"))

#x = tf.ones((3, 3))
#y = model(x)
#model.build()
print(model.summary())
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0) ,  # Optimizer
    # Loss function to minimize
    loss=tf.keras.losses.mean_squared_error,
    # List of metrics to monitor
    metrics=[tf.keras.metrics.RootMeanSquaredError()])


# split
X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_outputs, test_size=0.20)
#print(inputs.shape)
#print(X_train.shape)
print("Fit model on training data")

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True)

dnn_model = KerasRegressor(build_fn=model,
                          epochs=200,
                          batch_size=20,
                          verbose=0,
                          validation_split = 0.1)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

custom_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)


create_benchmark(data_inputs, data_outputs, new_model_list=[dnn_model], new_model_name_list=['DNN'], custom_scorer_list=[custom_scorer],
                 custom_scorer_name_list=['mape'], num_cv_fold=2)




#history = model.fit(
#    X_train,
#    y_train,
#    batch_size=80,
#    epochs=100,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
#    validation_data=(X_test, y_test),
#)



exit()






print('STOP')

create_benchmark(inputs, outputs, new_model_list=[dnn_model], new_model_name_list=['DNN'], custom_scorer_list=[custom_scorer],
                 custom_scorer_name_list=['mape'], num_cv_fold=2)

