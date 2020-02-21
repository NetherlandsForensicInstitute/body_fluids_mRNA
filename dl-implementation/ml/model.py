import logging
import os
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
from keras.layers import Dense, Dropout
from tensorflow import Tensor
from tensorflow.contrib.labeled_tensor.python.ops.core import Scalar

from ml.generator import EvalGenerator


logger = logging.getLogger('main')


def build_model(units: int, n_classes: int, n_features: int) -> Model:
    """
    Builds deep learning model

    :param units: (relative) number of units
    :param n_classes number of classes
    :param n_features: number of features
    :return: a keras model
    """
    # set drop out
    drop = 0.05

    # inout shape
    x = Input(shape=(n_features, ))
    # flatten input shape (i.e. remove the ,1)
    # first dense (hidden) layer
    cnn = Dense(units//4, activation="sigmoid")(x)
    # dropout
    cnn = Dropout(rate=drop)(cnn)
    # second dense (hidden) layer
    cnn = Dense(units, activation="sigmoid")(cnn)

    # output layer (corresponding to the number of classes)
    y = Dense(n_classes, activation="sigmoid")(cnn)

    # define inputs and outputs of the model
    model = Model(inputs=x, outputs=y)

    return model


def compile_model(model: Model, optimizer: str = "adam", loss: str = "binary_crossentropy") -> None:
    """
    compile a keras model using an optimizer and a loss function

    :param model: a keras model
    :param optimizer: a string or optimizer class that is supported by keras
    :param loss: a string or loss class that is supported by keras
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=[_accuracy_em])


def create_callbacks(batch_size: int, generator: EvalGenerator, log_dir: str,) -> list:
    """
    create callbacks to use in model.fit

    :param batch_size: batch size used for training the model
    :param generator: data generator
    :param log_dir: directory which is used for the logging
    :return: a list of callbacks
    """
    # create callbacks
    callbacks = [TensorBoard(log_dir=log_dir, batch_size=batch_size),
                 MetricsPerType(generator),
                 ModelCheckpoint(filepath=os.path.join(log_dir, 'model_weights_{epoch:02d}.hdf5'),
                                 save_best_only=False, save_weights_only=True)]

    return callbacks


def _accuracy_exact_match(y_true: Tensor, y_pred: Tensor, threshold: float = .5) -> Scalar:
    """
    Custom keras metric that mirrors the sklearn.metrics.accuracy_score, that is only samples that have the correct
    labels for each class are scored as 1. If not the sample is scored as 0.

    From: https://stackoverflow.com/questions/46799261/how-to-create-an-exact-match-eval-metric-op-for-tensorflow

    :param y_true: Tensor with the the true labels
    :param y_pred: Tensor with the predicted labels
    :param threshold: Threshold  used to classify a prediction as 1/0
    :return: float that represents the accuracy
    """
    # check if prediction are above threshold
    predictions = tf.to_float(tf.greater_equal(y_pred, threshold))
    # check if predictions match ground truth
    pred_match = tf.equal(predictions, tf.round(y_true))
    # reduce to mean
    exact_match = tf.reduce_min(tf.to_float(pred_match), axis=1)

    return exact_match


def _accuracy_em(*args) -> Scalar:
    """
    wrapper for _accuracy_exact_match

    :param args: input from metric evaluation provided by keras
    :return: float that represents the accuracy
    """
    return tf.reduce_mean(_accuracy_exact_match(*args))


class MetricsPerType(Callback):
    """
    Callback that computes metrics for each sample group (single/augmented/mixture)
    """
    def __init__(self, eval_generator: EvalGenerator, threshold: float = .5):
        object.__init__(self)
        self.eval_generator = eval_generator
        self.threshold = threshold
        self.y_pred = {}
        self.y_true = {}

    def on_train_end(self, logs={}) -> None:
        """
        calculate metrics at the and of training

        :param logs: logs from training
        """
        for sample_group, sample_types, index, in self.eval_generator.indexes:
            # select correct sample(s)
            samples = self.eval_generator.__getattribute__(sample_group)[sample_types][index]

            # average samples if multiple
            fin_sample = np.mean(samples, 0) if len(samples.shape) == 2 else samples
            # convert according to cut-off
            if self.eval_generator.cut_off:
                fin_sample = fin_sample > self.eval_generator.cut_off
            else:
                fin_sample /= 1000

            # init y
            y = np.zeros(self.eval_generator.n_classes)
            # Store class
            if sample_types:
                if sample_group == 'single':
                    sample_types = [sample_types]
                else:
                    sample_types = sample_types.split("+")
                for sample_type_idx in self.eval_generator.encoder.transform(sample_types):
                    y[sample_type_idx] = 1

            # predict
            y_pred = self.model.predict(np.expand_dims(fin_sample, 0))

            # store per group
            if sample_group not in self.y_pred:
                self.y_pred[sample_group] = [y_pred]
                self.y_true[sample_group] = [y]
            else:
                self.y_pred[sample_group].append(y_pred)
                self.y_true[sample_group].append(y)

        for i, sample_group in enumerate(self.y_true.keys()):
            # extract matrices and print metrics
            y_pred_mat, y_true_mat = self._convert_and_compute_metrics(sample_group)

            # store for total
            if i == 0:
                y_true_tot, y_pred_tot = y_true_mat, y_pred_mat
            else:
                y_true_tot = np.append(y_true_tot, y_true_mat, 0)
                y_pred_tot = np.append(y_pred_tot, y_pred_mat, 0)

        # calculate accuracy
        acc_tot = accuracy_score(y_true_tot, y_pred_tot)
        # print metrics
        logger.info(f'==Report for all sample groups==')
        logger.info(f'accuracy: {acc_tot:.2f}')
        logger.info(classification_report(y_true_tot, y_pred_tot, target_names=self.eval_generator.classes))

    def _convert_and_compute_metrics(self, sample_group: str) -> Tuple[np.array, np.array]:
        """
        Select sample group and extract all samples, compute metrics and return y_true and y_pred as numpy array

        :param sample_group: name of sample group
        :return: a numpy array with the predictions (binary), a numpy array with the ground truth
        """
        # convert y_true to numpy array
        y_true_mat = np.array(self.y_true[sample_group])
        # convert y_pred to numpy array and check if above threshold
        y_pred_mat = np.squeeze(np.array(self.y_pred[sample_group]) >= self.threshold)
        # calculate accuracy
        acc = accuracy_score(y_true_mat, y_pred_mat)
        # print metrics
        logger.info(f'==Report for {sample_group}==')
        logger.info(f'accuracy: {acc:.2f}')
        logger.info(classification_report(y_true_mat, y_pred_mat, target_names=self.eval_generator.classes))

        return y_pred_mat, y_true_mat
