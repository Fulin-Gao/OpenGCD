from OWR.methods.open_set_recognition.mi_calibration import mic
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from torch.autograd import Variable
import torch.optim as optim
import tensorflow as tf
import numpy as np
import torch
import copy
import os


def DNN(x_train, y_train, num_known_classes, args):
    # mic
    mixed_y = mic(x_train, y_train, num_known_classes)
    mixed_x = np.expand_dims(x_train, -1)
    idx = np.random.permutation(len(y_train))
    mixed_x = tf.convert_to_tensor(mixed_x[idx])
    mixed_y = tf.convert_to_tensor(mixed_y[idx])

    # creating model
    input_shape = mixed_x.shape[1:]
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            layers.Conv1D(filters=32, kernel_size=2, activation="relu", kernel_regularizer=regularizers.l1(0.01)),  # 32
            layers.Conv1D(filters=32, kernel_size=2, activation="relu", kernel_regularizer=regularizers.l1(0.01)),  # 32
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dropout(0.4),  
            layers.Dense(128, activation="relu"), 
            layers.Dropout(0.4),  
            layers.Dense(num_known_classes),
            layers.Softmax()
        ]
    )

    # lose
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.0,
        name="categorical_crossentropy"
    )

    # training
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.lr,  
        decay_steps=1000, 
        decay_rate=0.90) 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=loss_fn, metrics=['accuracy'])
    model.fit(mixed_x, mixed_y, epochs=args.epochs, batch_size=args.batch_size, verbose=1, validation_split=0.1)

    return model


def head(x_train, y_train, x_test, num_known_class, args, phase='1st', model=None):
    """
    :param x_train:               input of training set
    :param y_train:               output of training set
    :param x_test:                input of test set
    :param num_known_class:       number of known classes
    :param phase:                 current phase
    :param args:                  various parameters
    :param model:                 trained model
    :return:                      prediction results and new models
    """
    tf.random.set_seed(0)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    if model is None:
        # train and predict
        net = DNN(x_train, y_train, num_known_class, args)
        pred_prob = net.predict(np.expand_dims(x_test, -1))
        pred_label = np.argmax(pred_prob, axis=1)
        model = net
    else:
        # predict
        pred_prob = model.predict(np.expand_dims(x_test, -1))
        pred_label = np.argmax(pred_prob, axis=1)
        model = model

    return pred_prob, pred_label, model
