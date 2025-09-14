import glob
import numpy as np
import os
from os import path
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from tensorflow import keras
import tensorflow as tf
import warnings
import pickle
from dataPre import loadCsv, dataset_pre
from fl_logger import fl_logger

warnings.filterwarnings("ignore", category=Warning)

# Enable eager execution
tf.config.run_functions_eagerly(True)

TL = 4

trainPath_201 = "data/NewData/UNSW_NB15_Train201.csv"
trainPath_202 = "data/NewData/UNSW_NB15_Train202.csv"
trainPath_203 = "data/NewData/UNSW_NB15_Train203.csv"
trainPath_204 = "data/NewData/UNSW_NB15_Train204.csv"
trainPath_205 = "data/NewData/UNSW_NB15_Train205.csv"

testPath_2 = 'data/UNSW_NB15_TestBin.csv'

trainData_201 = loadCsv(trainPath_201)
trainData_202 = loadCsv(trainPath_202)
trainData_203 = loadCsv(trainPath_203)
trainData_204 = loadCsv(trainPath_204)
trainData_205 = loadCsv(trainPath_205)

testData_2 = loadCsv(testPath_2)

trainData01_scaler = trainData_201[:, 0:196]
trainData02_scaler = trainData_202[:, 0:196]
trainData03_scaler = trainData_203[:, 0:196]
trainData04_scaler = trainData_204[:, 0:196]
trainData05_scaler = trainData_205[:, 0:196]

testData_scaler = testData_2[:, 0:196]

# Load the single, consistent scaler fitted on training features
with open("CentralServer/scaler.pkl", "rb") as _f:
    scaler = pickle.load(_f)

trainData01_scaler = scaler.transform(trainData01_scaler)
trainData02_scaler = scaler.transform(trainData02_scaler)
trainData03_scaler = scaler.transform(trainData03_scaler)
trainData04_scaler = scaler.transform(trainData04_scaler)
trainData05_scaler = scaler.transform(trainData05_scaler)
testData_scaler = scaler.transform(testData_scaler)

x_train01 = dataset_pre(trainData01_scaler, TL)
x_train01 = np.reshape(x_train01, (-1, TL, 196))
x_train02 = dataset_pre(trainData02_scaler, TL)
x_train02 = np.reshape(x_train02, (-1, TL, 196))
x_train03 = dataset_pre(trainData03_scaler, TL)
x_train03 = np.reshape(x_train03, (-1, TL, 196))
x_train04 = dataset_pre(trainData04_scaler, TL)
x_train04 = np.reshape(x_train04, (-1, TL, 196))
x_train05 = dataset_pre(trainData05_scaler, TL)
x_train05 = np.reshape(x_train05, (-1, TL, 196))

x_test = dataset_pre(testData_scaler, TL)
x_test = np.reshape(x_test, (-1, TL, 196))

# Label
y_train01 = trainData_201[:,196]
y_train02 = trainData_202[:,196]
y_train03 = trainData_203[:,196]
y_train04 = trainData_204[:,196]
y_train05 = trainData_205[:,196]
y_test = testData_2[:,196]

shape = np.size(x_train01, axis=2)

def nids_model01(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
        # Recompile the model after loading
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    history = model.fit(x_train01, y_train01, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)
    
    # Log training metrics
    for epoch in range(serverepochs):
        fl_logger.log_training_step(
            server_name="Server1",
            epoch=epoch,
            accuracy=history.history['acc'][epoch] * 100,
            loss=history.history['loss'][epoch],
            val_accuracy=history.history['val_acc'][epoch] * 100 if 'val_acc' in history.history else None,
            val_loss=history.history['val_loss'][epoch] if 'val_loss' in history.history else None
        )

    m = model.get_weights()
    # Save weights as a list (not numpy array) to handle different shapes
    import pickle
    with open('Server/Server1.pkl', 'wb') as f:
        pickle.dump(m, f)
    return model

def nids_model02(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
        # Recompile the model after loading
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    history = model.fit(x_train02, y_train02, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)
    
    # Log training metrics
    for epoch in range(serverepochs):
        fl_logger.log_training_step(
            server_name="Server2",
            epoch=epoch,
            accuracy=history.history['acc'][epoch] * 100,
            loss=history.history['loss'][epoch],
            val_accuracy=history.history['val_acc'][epoch] * 100 if 'val_acc' in history.history else None,
            val_loss=history.history['val_loss'][epoch] if 'val_loss' in history.history else None
        )

    m = model.get_weights()
    # Save weights as a list (not numpy array) to handle different shapes
    import pickle
    with open('Server/Server2.pkl', 'wb') as f:
        pickle.dump(m, f)
    return model

def nids_model03(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
        # Recompile the model after loading
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    history = model.fit(x_train03, y_train03, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)
    
    # Log training metrics
    for epoch in range(serverepochs):
        fl_logger.log_training_step(
            server_name="Server3",
            epoch=epoch,
            accuracy=history.history['acc'][epoch] * 100,
            loss=history.history['loss'][epoch],
            val_accuracy=history.history['val_acc'][epoch] * 100 if 'val_acc' in history.history else None,
            val_loss=history.history['val_loss'][epoch] if 'val_loss' in history.history else None
        )

    m = model.get_weights()
    # Save weights as a list (not numpy array) to handle different shapes
    import pickle
    with open('Server/Server3.pkl', 'wb') as f:
        pickle.dump(m, f)
    return model


def nids_model04(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
        # Recompile the model after loading
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    history = model.fit(x_train04, y_train04, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)
    
    # Log training metrics
    for epoch in range(serverepochs):
        fl_logger.log_training_step(
            server_name="Server4",
            epoch=epoch,
            accuracy=history.history['acc'][epoch] * 100,
            loss=history.history['loss'][epoch],
            val_accuracy=history.history['val_acc'][epoch] * 100 if 'val_acc' in history.history else None,
            val_loss=history.history['val_loss'][epoch] if 'val_loss' in history.history else None
        )

    m = model.get_weights()
    # Save weights as a list (not numpy array) to handle different shapes
    import pickle
    with open('Server/Server4.pkl', 'wb') as f:
        pickle.dump(m, f)
    return model

def nids_model05(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
        # Recompile the model after loading
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    history = model.fit(x_train05, y_train05, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)
    
    # Log training metrics
    for epoch in range(serverepochs):
        fl_logger.log_training_step(
            server_name="Server5",
            epoch=epoch,
            accuracy=history.history['acc'][epoch] * 100,
            loss=history.history['loss'][epoch],
            val_accuracy=history.history['val_acc'][epoch] * 100 if 'val_acc' in history.history else None,
            val_loss=history.history['val_loss'][epoch] if 'val_loss' in history.history else None
        )

    m = model.get_weights()
    # Save weights as a list (not numpy array) to handle different shapes
    import pickle
    with open('Server/Server5.pkl', 'wb') as f:
        pickle.dump(m, f)

    return model

def load_models():
    arr = []
    models = glob.glob("Server/*.pkl")
    for i in models:
        import pickle
        with open(i, 'rb') as f:
            weights = pickle.load(f)
        arr.append(weights)

    return arr

def fl_average():
    arr = load_models()
    # Average weights layer by layer
    fl_avg = []
    for layer_idx in range(len(arr[0])):
        layer_weights = [model[layer_idx] for model in arr]
        avg_layer = np.mean(layer_weights, axis=0)
        fl_avg.append(avg_layer)
    
    return fl_avg

def build_model(avg):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
    model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    model.set_weights(avg)
    print("FL Model Ready!")

    return model

def evaluate_model(model, x_test, y_test):
    print('Test Num:', len(y_test))
    score = model.evaluate(x_test, y_test, batch_size=200000, verbose=0)
    print('Score:', score)
    return score

def save_fl_model(model):
    model.save("CentralServer/fl_model.h5")

def model_fl():
    avg = fl_average()
    model = build_model(avg)
    score = evaluate_model(model, x_test, y_test)
    
    # Log federated aggregation results
    fl_logger.log_federated_aggregation(
        epoch=len(glob.glob("logs/combined_training.json")) if os.path.exists("logs/combined_training.json") else 0,
        test_accuracy=score[1] * 100,  # score[1] is accuracy
        test_loss=score[0],  # score[0] is loss
        server_count=5
    )
    
    save_fl_model(model)

fl_epochs = 300

# Log system start
fl_logger.log_system_status("started", f"Starting federated learning with {fl_epochs} epochs")

for i in range(fl_epochs):
    print(f'Starting Epoch {i+1}/{fl_epochs}')
    
    # Log epoch start
    fl_logger.log_system_status("training", f"Training epoch {i+1}")

    model1 = nids_model01(shape, 500, 1)
    model2 = nids_model02(shape, 500, 1)
    model3 = nids_model03(shape, 500, 1)
    model4 = nids_model04(shape, 500, 1)
    model5 = nids_model05(shape, 500, 1)
    model_fl()
    
    print('Epoch:', i+1)
    
    # Log epoch completion
    fl_logger.log_system_status("completed", f"Completed epoch {i+1}")

    K.clear_session()

# Log system completion
fl_logger.log_system_status("completed", f"Federated learning completed after {fl_epochs} epochs")