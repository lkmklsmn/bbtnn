#!/usr/bin/env python
# coding: utf-8

from ivis.nn.losses import triplet_loss, is_categorical, is_multiclass, is_hinge
from ivis.nn.network import triplet_network, base_network
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, AlphaDropout, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
import io
import numpy as np
import json
import os
import shutil
import multiprocessing
from scipy.sparse import issparse
import bbknn
from tensorflow.keras import backend as K
import tensorflow as tf
from ivis import Ivis
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder


class KnnTripletGenerator(Sequence):

    def __init__(self, X, batch, neighbour_matrix, batch_size=32):
        self.X = X
        self.batch = batch
        self.neighbour_matrix = neighbour_matrix
        self.batch_size = batch_size
        self.placeholder_labels = np.empty(batch_size, dtype=np.uint8)

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.X.shape[0]))
        placeholder_labels = self.placeholder_labels[:len(batch_indices)]
        triplet_batch = [self.knn_triplet_from_neighbour_list(row_index, neighbour_matrix) for row_index in batch_indices]

        triplet_batch = np.array(triplet_batch)

        return([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]], placeholder_labels)

    def knn_triplet_from_neighbour_list(self, row_index, neighbour_matrix):
        """ A random (unweighted) positive example chosen. """
        triplets = []
        anchor = row_index
        positive = int(neighbour_matrix[row_index, np.random.randint(0, neighbour_matrix.shape[1], 1)])
        negative = int(np.random.randint(0, neighbour_matrix.shape[0], 1))
        triplets += [self.X[anchor], self.X[positive], self.X[negative]]
        return triplets



class LabeledKnnTripletGenerator(Sequence):

    def __init__(self, X, Y, batch, neighbour_matrix, batch_size=32):
        self.X, self.Y = X, Y
        self.batch = batch
        self.neighbour_matrix = neighbour_matrix
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.X.shape[0]))

        label_batch = self.Y[batch_indices]
        triplet_batch = [self.knn_triplet_from_neighbour_list(row_index, self.neighbour_matrix[row_index])
                         for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)

        return tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]), tuple([np.array(label_batch), np.array(label_batch)])

    def knn_triplet_from_neighbour_list(self, row_index, neighbour_list):
        """ A random (unweighted) positive example chosen. """
        triplets = []

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(neighbour_list)

        # Take a random non-neighbour as negative
        # Pick a random index until one fits constraint. An optimization.
        negative_ind = np.random.randint(0, self.X.shape[0])
        while negative_ind in neighbour_list:
            negative_ind = np.random.randint(0, self.X.shape[0])

        triplets += [self.X[row_index],
                     self.X[neighbour_ind],
                     self.X[negative_ind]]
        return triplets


def generator_from_index(X, Y, batch, k = 5, batch_size = 16, search_k=-1, verbose=1):
        if k >= X.shape[0] - 1:
                raise Exception('''k value greater than or equal to (num_rows - 1)(k={}, rows={}). Lower k to a smaller value.'''.format(k, X.shape[0]))

        if batch_size > X.shape[0]:
                raise Exception('''batch_size value larger than num_rows in dataset (batch_size={}, rows={}). Lower batch_size to a smaller value.'''.format(batch_size, X.shape[0]))

        knn_distances, knn_indices=bbknn.get_graph(pca=X, batch_list = batch, neighbors_within_batch=3, n_pcs=50, approx=True, metric="euclidean", use_faiss=True, n_trees=10)
        neighbour_matrix = knn_indices

        if Y is None:
            return KnnTripletGenerator(X = X, batch = batch, neighbour_matrix = neighbour_matrix, batch_size=batch_size)
        else:
            return LabeledKnnTripletGenerator(X = X, Y = Y, batch = batch, neighbour_matrix = neighbour_matrix, batch_size=batch_size)



class BBTNN(BaseEstimator):
    def __init__(self, embedding_dims=2, k=150, distance='pn', batch_size=128, batch=2,
                 epochs=1000, n_epochs_without_progress=50,
                 margin=1, ntrees=50, search_k=-1,
                 model='default',supervision_metric='sparse_categorical_crossentropy',
                 supervision_weight=0.5,
                 callbacks=[], eager_execution=False, verbose=1):

        self.embedding_dims = embedding_dims
        self.k = k
        self.distance = distance
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_epochs_without_progress = n_epochs_without_progress
        self.margin = margin
        self.ntrees = ntrees
        self.search_k = search_k
        self.model_def = model
        self.model_ = None
        self.encoder = None
        self.supervision_metric = supervision_metric
        self.supervision_weight = supervision_weight
        self.loss_history_ = []
        self.callbacks = callbacks
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback = callback.register_ivis_model(self)
        self.eager_execution = eager_execution
        if not eager_execution:
            tf.compat.v1.disable_eager_execution()
        self.verbose = verbose

    def __getstate__(self):
        """ Return object serializable variable dict """

        state = dict(self.__dict__)
        if 'model_' in state:
            state['model_'] = None
        if 'encoder' in state:
            state['encoder'] = None
        if 'supervised_model_' in state:
            state['supervised_model_'] = None
        if 'callbacks' in state:
            state['callbacks'] = []
        if not isinstance(state['model_def'], str):
            state['model_def'] = None
        return state

    def _fit(self, batch, X, Y=None, shuffle_mode=True):

        datagen = generator_from_index(X, Y, batch=batch,
                                       k=self.k,
                                       batch_size=self.batch_size,
                                       search_k=self.search_k,
                                       verbose=self.verbose)

        loss_monitor = 'loss'
        try:
            triplet_loss_func = triplet_loss(distance=self.distance,
                                             margin=self.margin)
        except KeyError:
            raise ValueError('Loss function `{}` not implemented.'.format(self.distance))

        if self.model_ is None:
            if type(self.model_def) is str:
                input_size = (X.shape[-1],)
                self.model_, anchor_embedding, _, _ = \
                    triplet_network(base_network(self.model_def, input_size),
                                    embedding_dims=self.embedding_dims)
            else:
                self.model_, anchor_embedding, _, _ = \
                    triplet_network(self.model_def,
                                    embedding_dims=self.embedding_dims)

            if Y is None:
                self.model_.compile(optimizer='adam', loss=triplet_loss_func)
            else:
                if is_categorical(self.supervision_metric):
                    if not is_multiclass(self.supervision_metric):
                        if not is_hinge(self.supervision_metric):
                            # Binary logistic classifier
                            if len(Y.shape) > 1:
                                self.n_classes = Y.shape[-1]
                            else:
                                self.n_classes = 1
                            supervised_output = Dense(self.n_classes, activation='sigmoid',
                                                      name='supervised')(anchor_embedding)
                        else:
                            # Binary Linear SVM output
                            if len(Y.shape) > 1:
                                self.n_classes = Y.shape[-1]
                            else:
                                self.n_classes = 1
                            supervised_output = Dense(self.n_classes, activation='linear',
                                                      name='supervised',
                                                      kernel_regularizer=regularizers.l2())(anchor_embedding)
                    else:
                        if not is_hinge(self.supervision_metric):
                            validate_sparse_labels(Y)
                            self.n_classes = len(np.unique(Y[Y != np.array(-1)]))
                            # Softmax classifier
                            supervised_output = Dense(self.n_classes, activation='softmax',
                                                      name='supervised')(anchor_embedding)
                        else:
                            self.n_classes = len(np.unique(Y, axis=0))
                            # Multiclass Linear SVM output
                            supervised_output = Dense(self.n_classes, activation='linear',
                                                      name='supervised',
                                                      kernel_regularizer=regularizers.l2())(anchor_embedding)
                else:
                    # Regression
                    if len(Y.shape) > 1:
                        self.n_classes = Y.shape[-1]
                    else:
                        self.n_classes = 1
                    supervised_output = Dense(self.n_classes, activation='linear',
                                              name='supervised')(anchor_embedding)

                supervised_loss = keras.losses.get(self.supervision_metric)
                if self.supervision_metric == 'sparse_categorical_crossentropy':
                    supervised_loss = semi_supervised_loss(supervised_loss)

                final_network = Model(inputs=self.model_.inputs,
                                      outputs=[self.model_.output,
                                               supervised_output])
                self.model_ = final_network
                self.model_.compile(
                    optimizer='adam',
                    loss={
                        'stacked_triplets': triplet_loss_func,
                        'supervised': supervised_loss
                         },
                    loss_weights={
                        'stacked_triplets': 1 - self.supervision_weight,
                        'supervised': self.supervision_weight})

                # Store dedicated classification model
                supervised_model_input = Input(shape=(X.shape[-1],))
                embedding = self.model_.layers[3](supervised_model_input)
                softmax_out = self.model_.layers[-1](embedding)

                self.supervised_model_ = Model(supervised_model_input, softmax_out)

        self.encoder = self.model_.layers[3]

        if self.verbose > 0:
            print('Training neural network')

        hist = self.model_.fit(
            datagen,
            epochs=self.epochs,
            callbacks=[callback for callback in self.callbacks] +
                      [EarlyStopping(monitor=loss_monitor,
                       patience=self.n_epochs_without_progress)],
            shuffle=shuffle_mode,
            workers=multiprocessing.cpu_count(),
            verbose=self.verbose)
        self.loss_history_ += hist.history['loss']


    def fit(self, X, Y=None, shuffle_mode=True):
        """Fit an ivis model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data to be embedded.
        Y : array, shape (n_samples)
            Optional array for supervised dimentionality reduction.

        Returns
        -------
        returns an instance of self
        """

        self._fit(X, Y, shuffle_mode)
        return self

    def fit_transform(self, X, Y=None, shuffle_mode=True):
        """Fit to data then transform

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data to be embedded.
        Y : array, shape (n_samples)
            Optional array for supervised dimentionality reduction.


        Returns
        -------
        X_new : transformed array, shape (n_samples, embedding_dims)
            Embedding of the new data in low-dimensional space.
        """

        self.fit(X, Y, shuffle_mode)
        return self.transform(X)

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.

        Returns
        -------
        X_new : array, shape (n_samples, embedding_dims)
            Embedding of the new data in low-dimensional space.
        """

        embedding = self.encoder.predict(X, verbose=self.verbose)
        return embedding

    def score_samples(self, X):
        """Passes X through classification network to obtain predicted
        supervised values. Only applicable when trained in
        supervised mode.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data to be passed through classification network.
        Returns
        -------
        X_new : array, shape (n_samples, embedding_dims)
            Softmax class probabilities of the data.
        """

        if self.supervised_model_ is None:
            raise Exception("Model was not trained in classification mode.")

        softmax_output = self.supervised_model_.predict(X, verbose=self.verbose)
        return softmax_output

def semi_supervised_loss(loss_function):
    def new_loss_function(y_true, y_pred):
        mask = tf.cast(~tf.math.equal(y_true, -1), tf.float32)
        y_true_pos = tf.nn.relu(y_true)
        loss = loss_function(y_true_pos, y_pred)
        masked_loss = loss * mask
        return masked_loss
    new_func = new_loss_function
    new_func.__name__ = loss_function.__name__
    return new_func

def validate_sparse_labels(Y):
    if not zero_indexed(Y):
        raise ValueError('Ensure that your labels are zero-indexed')
    if not consecutive_indexed(Y):
        raise ValueError('Ensure that your labels are indexed consecutively')

def zero_indexed(Y):
    if min(abs(Y)) != 0:
        return False
    return True

def consecutive_indexed(Y):
    """ Assumes that Y is zero-indexed. """
    n_classes = len(np.unique(Y[Y != np.array(-1)]))
    if max(Y) >= n_classes:
        return False
    return True

def unsupervised_bbtnn(
    pca,
    batch,
    neighbors_within_batch=5,
    n_pcs=20,
    approx=True,
    metric="euclidean",
    use_faiss=True,
    n_trees=10,
    model='maaten',
    verbose=1,
    k=15,
    distance='pn',
    batch_size=128,
    epochs=100,
    n_epochs_without_progress=5):

    knn_distances, knn_indices=bbknn.get_graph(pca=pca, batch_list = batch, neighbors_within_batch=neighbors_within_batch, n_pcs=n_pcs, approx=approx, metric=metric, use_faiss=use_faiss, n_trees=n_trees)
    neighbour_matrix = knn_indices
    bbtnn_model = BBTNN(model=model, verbose=verbose, k=k, distance=distance, batch_size=batch_size, batch=batch, epochs=epochs, n_epochs_without_progress=n_epochs_without_progress)
    embeddings = bbtnn_model.fit_transform(pca, None)
    return(embeddings)
