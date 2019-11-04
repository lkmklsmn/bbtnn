#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import bbknn
from ivis.nn.losses import triplet_loss
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
import io
import numpy as np
import json
import os
import shutil
from scipy.sparse import issparse
from tensorflow.keras import backend as K
import tensorflow as tf
from ivis import Ivis
from tensorflow import keras

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


def generator_from_index(X, batch, k = 5, batch_size = batch_size, search_k=-1, verbose=1):
        if k >= X.shape[0] - 1:
                raise Exception('''k value greater than or equal to (num_rows - 1)(k={}, rows={}). Lower k to a smaller value.'''.format(k, X.shape[0]))

        if batch_size > X.shape[0]:
                raise Exception('''batch_size value larger than num_rows in dataset (batch_size={}, rows={}). Lower batch_size to a smaller value.'''.format(batch_size, X.shape[0]))

        knn_distances, knn_indices=bbknn.get_graph(pca=X, batch_list = batch, neighbors_within_batch=k, n_pcs=50, approx=True, metric="euclidean", use_faiss=True, n_trees=10)

        neighbour_matrix = knn_indices
        return KnnTripletGenerator(X = X, batch = batch, neighbour_matrix = neighbour_matrix, batch_size=batch_size)


class Ivis(BaseEstimator):
    def __init__(self, embedding_dims=2, k=20, distance='pn', batch_size=32,
                 epochs=100, n_epochs_without_progress=5,
                 margin=1, ntrees=50, search_k=-1,
                 model='default',
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

    def _fit(self, X, Y=None, shuffle_mode=True):

        datagen = generator_from_index(X, batch=Y,
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
                self.model_, anchor_embedding, _, _ = triplet_network(base_network(self.model_def, input_size), embedding_dims=self.embedding_dims)
            else:
                self.model_, anchor_embedding, _, _ = triplet_network(self.model_def, embedding_dims=self.embedding_dims)

        self.model_.compile(optimizer='adam', loss=triplet_loss_func)

        self.encoder = self.model_.layers[3]

        if self.verbose > 0:
            print('Training neural network')

        hist = self.model_.fit_generator(
            datagen,
            steps_per_epoch=X.shape[0] // self.batch_size,
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
