#!/usr/bin/env python
# coding: utf-8

import bbknn
import numpy as np
import scanpy as sc
from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, AlphaDropout, Lambda
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.utils import Sequence
from ivis.nn.losses import triplet_loss, is_categorical, is_multiclass, is_hinge
import pandas as pd
import sys
from scipy.spatial import cKDTree
import random
from sklearn.base import BaseEstimator
from ivis.nn.network import triplet_network
from ivis.nn.callbacks import ModelCheckpoint
from ivis.nn.losses import triplet_loss, is_categorical, is_multiclass, is_hinge
from ivis.nn.losses import semi_supervised_loss, validate_sparse_labels
import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
import json
import os
import shutil
import multiprocessing
import tensorflow as tf
import platform

def base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    inputs = Input(shape=input_shape)
    n_dim = round(0.75 * input_shape[0])
    x = Dense(n_dim, activation='selu',
              kernel_initializer='lecun_normal')(inputs)
    x = AlphaDropout(0.25)(x)
    x = Dense(n_dim, activation='selu',
              kernel_initializer='lecun_normal')(x)
    x = AlphaDropout(0.25)(x)
    x = Dense(n_dim, activation='selu', kernel_initializer='lecun_normal')(x)
    return Model(inputs, x)

def extract_knn(X, batch_list, k, verbose = True):
    if verbose:
        print("Calculating KNN graph")

    knn_distances, knn_indices = bbknn.get_graph(
                pca = X, batch_list = batch_list,
                neighbors_within_batch = k, n_pcs = X.shape[1],
                approx = True, metric = "euclidean", use_faiss = False, n_trees = 100)

    return(knn_indices)

def generator_from_index(X, batch_list, k, batch_size, search_k=-1,
                         precompute=True, verbose=1):
    if k >= X.shape[0] - 1:
        raise Exception('''k value greater than or equal to (num_rows - 1)
                        (k={}, rows={}). Lower k to a smaller
                        value.'''.format(k, X.shape[0]))
    if batch_size > X.shape[0]:
        raise Exception('''batch_size value larger than num_rows in dataset
                        (batch_size={}, rows={}). Lower batch_size to a
                        smaller value.'''.format(batch_size, X.shape[0]))

    neighbour_matrix = extract_knn(X = X, batch_list = batch_list, k = k)

    indices_by_batch = []
    names_indices_by_batch = []
    for batch in batch_list.unique():
        names_indices_by_batch.append(batch)
        indices_by_batch.append(np.where(batch_list == batch)[0])

    batch_list_num =[]
    for i in batch_list:
        batch_list_num.append(names_indices_by_batch.index(i))

    return KnnTripletGenerator(X, neighbour_matrix, batch_list = batch_list_num, indices_by_batch = indices_by_batch, batch_size=batch_size)

class KnnTripletGenerator(Sequence):

    def __init__(self, X, neighbour_matrix, batch_list, batch_size=32):
        self.X = X
        self.neighbour_matrix = neighbour_matrix
        self.batch_size = batch_size
        self.batch_list = batch_list
        self.placeholder_labels = np.empty(batch_size, dtype=np.uint8)

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.X.shape[0]))

        placeholder_labels = self.placeholder_labels[:len(batch_indices)]
        triplet_batch = [self.knn_triplet_from_neighbour_list(row_index, self.neighbour_matrix[row_index], self.batch_list[row_index])
                         for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]
        triplet_batch = np.array(triplet_batch)

        return tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]), placeholder_labels

    def knn_triplet_from_neighbour_list(self, row_index, neighbour_list, rowindex_batch):
        """ A random (unweighted) positive example chosen. """
        triplets = []

        # Take a random neighbour as positive
        neighbour_ind = np.random.choice(neighbour_list)

        # Take a random non-neighbour as negative
        # Pick a random index until one fits constraint. An optimization.
        negative_ind = np.random.choice(self.indices_by_batch[rowindex_batch])

        triplets += [self.X[row_index], self.X[neighbour_ind],
                     self.X[negative_ind]]
        return triplets

class BBTNN(BaseEstimator):
    """Ivis is a technique that uses an artificial neural network for
    dimensionality reduction, often useful for the purposes of visualization.
    The network trains on triplets of data-points at a time and pulls positive
    points together, while pushing more distant points away from each other.
    Triplets are sampled from the original data using KNN aproximation using
    the Annoy library.
    :param int embedding_dims: Number of dimensions in the embedding space
    :param int k: The number of neighbours to retrieve for each point.
        Must be less than one minus the number of rows in the dataset.
    :param str distance: The loss function used to train the neural network.
        One of "pn", "euclidean", "manhattan_pn", "manhattan", "chebyshev",
        "chebyshev_pn", "softmax_ratio_pn", "softmax_ratio", "cosine",
        "cosine_pn".
    :param int batch_size: The size of mini-batches used during gradient
        descent while training the neural network. Must be less than the
        num_rows in the dataset.
    :param int epochs: The maximum number of epochs to train the model for.
        Each epoch the network will see a triplet based on each data-point
        once.
    :param int n_epochs_without_progress: After n number of epochs without an
        improvement to the loss, terminate training early.
    :param float margin: The distance that is enforced between points by the
        triplet loss functions.
    :param int ntrees: The number of random projections trees built by Annoy to
        approximate KNN. The more trees the higher the memory usage, but the
        better the accuracy of results.
    :param int search_k: The maximum number of nodes inspected during a nearest
        neighbour query by Annoy. The higher, the more computation time
        required, but the higher the accuracy. The default is n_trees * k,
        where k is the number of neighbours to retrieve. If this is set too
        low, a variable number of neighbours may be retrieved per data-point.
    :param bool precompute: Whether to pre-compute the nearest neighbours.
        Pre-computing is a little faster, but requires more memory. If memory
        is limited, try setting this to False.
    :param str model: str or keras.models.Model. The keras model to train using
        triplet loss. If a model object is provided, an embedding layer of size
        'embedding_dims' will be appended to the end of the network.
        If a string, a pre-defined network by that name will be used. Possible
        options are: 'szubert', 'hinton', 'maaten'. By default the 'szubert'
        network will be created, which is a selu network composed of 3 dense
        layers of 128 neurons each, followed by an embedding layer of size
        'embedding_dims'.
    :param str supervision_metric: str or function. The supervision metric to
        optimize when training keras in supervised mode. Supports all of the
        classification or regression losses included with keras, so long as
        the labels are provided in the correct format. A list of keras' loss
        functions can be found at https://keras.io/losses/ .
    :param float supervision_weight: Float between 0 and 1 denoting the
        weighting to give to classification vs triplet loss when training
        in supervised mode. The higher the weight, the more classification
        influences training. Ignored if using Ivis in unsupervised mode.
    :param str annoy_index_path: The filepath of a pre-trained annoy index file
        saved on disk. If provided, the annoy index file will be used.
        Otherwise, a new index will be generated and saved to disk in the
        current directory as 'annoy.index'.
    :param list[keras.callbacks.Callback] callbacks: List of keras Callbacks to
        pass model during training, such as the TensorBoard callback. A set of
        ivis-specific callbacks are provided in the ivis.nn.callbacks module.
    :param bool build_index_on_disk: Whether to build the annoy index directly
        on disk. Building on disk should allow for bigger datasets to be indexed,
        but may cause issues. If None, on-disk building will be enabled for Linux,
        but not Windows due to issues on Windows.
    :param int verbose: Controls the volume of logging output the model
        produces when training. When set to 0, silences outputs, when above 0
        will print outputs.
    """

    def __init__(self, embedding_dims=2, k=150, distance='pn', batch_size=128,
                 epochs=1000, n_epochs_without_progress=20,
                 margin=1, ntrees=50, search_k=-1,
                 precompute=True, model='szubert',
                 supervision_metric='sparse_categorical_crossentropy',
                 supervision_weight=0.5, annoy_index_path=None,
                 callbacks=[], build_index_on_disk=None, verbose=1):

        self.embedding_dims = embedding_dims
        self.k = k
        self.distance = distance
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_epochs_without_progress = n_epochs_without_progress
        self.margin = margin
        self.ntrees = ntrees
        self.search_k = search_k
        self.precompute = precompute
        self.model_def = model
        self.model_ = None
        self.encoder = None
        self.supervision_metric = supervision_metric
        self.supervision_weight = supervision_weight
        self.supervised_model_ = None
        self.loss_history_ = []
        self.annoy_index_path = annoy_index_path
        self.callbacks = callbacks
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback = callback.register_ivis_model(self)
        if build_index_on_disk is None:
            self.build_index_on_disk = True if platform.system() != 'Windows' else False
        else:
            self.build_index_on_disk = build_index_on_disk
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

    def _fit(self, X, batch_list, shuffle_mode=True):

        datagen = generator_from_index(X, batch_list,
                                       k=self.k,
                                       batch_size=self.batch_size,
                                       search_k=self.search_k)

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
                    triplet_network(base_network(input_size),
                                    embedding_dims=self.embedding_dims)
            else:
                self.model_, anchor_embedding, _, _ = \
                    triplet_network(self.model_def,
                                    embedding_dims=self.embedding_dims)


        self.model_.compile(optimizer='adam', loss=triplet_loss_func)

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

    def fit(self, X, batch_list, shuffle_mode=True):
        """Fit an ivis model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data to be embedded.
        Y : array, shape (n_samples)
            Optional array for supervised dimentionality reduction.
            If Y contains -1 labels, and 'sparse_categorical_crossentropy'
            is the loss function, semi-supervised learning will be used.
        Returns
        -------
        returns an instance of self
        """

        self._fit(X, batch_list, shuffle_mode)
        return self

    def fit_transform(self, X, Y=None, shuffle_mode=True):
        """Fit to data then transform
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data to be embedded.
        Y : array, shape (n_samples)
            Optional array for supervised dimentionality reduction.
            If Y contains -1 labels, and 'sparse_categorical_crossentropy'
            is the loss function, semi-supervised learning will be used.
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
