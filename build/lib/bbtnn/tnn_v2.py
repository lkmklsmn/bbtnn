import numpy as np
import scanpy as sc
import pandas as pd

from annoy import AnnoyIndex

from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, AlphaDropout, Lambda
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.utils import Sequence

import random
from ivis.nn.losses import *
from ivis.nn.network import triplet_network, base_network
from ivis.nn.callbacks import ModelCheckpoint

from scipy.sparse import issparse
from scipy.spatial import cKDTree

from sklearn.base import BaseEstimator
import json
import os
import shutil
import multiprocessing
import platform

from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from intervaltree import IntervalTree
import operator

import itertools
from annoy import AnnoyIndex
import multiprocessing
from multiprocessing import Pool
cpus = multiprocessing.cpu_count()
from functools import partial

import networkx as nx

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


def generator_from_index(adata, batch_name = 'batch', k = 20, batch_size = 32, search_k=-1,
                         precompute=True, verbose=1):

    if verbose > 0:
        print('Calculate MNN pairs')
    mnn_matches = calculate_mnns(adata, batch_name = batch_name, knn = k)
    if verbose > 0:
        print('Reformat MNN pairs')
    mnn_neighbour_list, cells_for_mnn = convert_mnns_pairs_to_list_3(mnn_matches, cell_names = adata.obs_names)

    if verbose > 0:
        print('Calculate KNN neighborhood')
    cells_for_knn = find_leftover_cells(adata, mnn_matches)
    X = adata[adata.obs_names.isin(cells_for_knn)].obsm["X_pca"]
    knn_matrix = calculate_knns(X)
    knn_neighbour_list = []
    for i in range(0, knn_matrix.shape[0]):
      knn_neighbour_list.append(list(knn_matrix[i,:]))

    if verbose > 0:
        print ('******Batches:'+ str(adata.obs[batch_name].unique()))
        print ('******Total number of cells:'+ str(adata.shape[0]))
        print ('******Number of cells for MNN:'+ str(len(cells_for_mnn)))
        print ('******Number of cells for KNN:'+ str(len(cells_for_knn)))

    final_neighbour_list = mnn_neighbour_list + knn_neighbour_list
    final_cells = cells_for_mnn + list(cells_for_knn)

    if verbose > 0:
        print('Sort cells')
    triplet_list = []
    for i in adata.obs_names:
      triplet_list.append(final_neighbour_list[final_cells.index(i)])
    #new_order = adata.obs.reindex(final_cells).index.tolist()
    #adata = adata[final_cells]
    #tmp = pd.Series(range(0, len(final_cells)), index = final_cells)[adata.obs_names]
    #final_neighbour_list = np.array(final_neighbour_list)[np.array(tmp)].tolist()

    return KnnTripletGenerator(X = adata.obsm["X_pca"], dictionary = triplet_list, batch_size = batch_size)


class KnnTripletGenerator(Sequence):

    def __init__(self, X, dictionary, batch_size=32):
        self.X = X
        self.batch_size = batch_size
        self.dictionary = dictionary
        self.placeholder_labels = np.empty(batch_size, dtype=np.uint8)
        self.num_cells = len(self.dictionary)

    def __len__(self):
        return int(np.ceil(len(self.dictionary) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.num_cells))

        triplet_batch = [self.knn_triplet_from_dictionary(row_index = row_index, neighbour_list = self.dictionary[row_index], num_cells = self.num_cells) for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)
        placeholder_labels = self.placeholder_labels[:triplet_batch.shape[0]]

        return tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]), placeholder_labels

    def knn_triplet_from_dictionary(self, row_index, neighbour_list, num_cells):
        """ A random (unweighted) positive example chosen. """
        triplets = []

        #print(type(cell_names))

        anchor = row_index
        positive = np.random.choice(neighbour_list)
        #positive = int(np.where(self.cell_names == positive)[0])
        negative = np.random.randint(self.num_cells)

        triplets += [self.X[anchor], self.X[positive],
                     self.X[negative]]

        return triplets


class TNN(BaseEstimator):

    def __init__(self, embedding_dims=2, k=150, distance='pn', batch_size=64,
                 epochs=1000, n_epochs_without_progress=20,
                 margin=1, ntrees=50, search_k=-1,
                 precompute=True,
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
        self.model_def = "dummy"
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

    def _fit(self, X, batch_name, Y=None, shuffle_mode=True):

        datagen = generator_from_index(X,
                                       batch_name = batch_name,
                                       k=self.k,
                                       batch_size=self.batch_size,
                                       search_k=self.search_k,
                                       precompute=True, verbose=1)

        loss_monitor = 'loss'
        try:
            triplet_loss_func = triplet_loss(distance=self.distance,
                                             margin=self.margin)
        except KeyError:
            raise ValueError('Loss function `{}` not implemented.'.format(self.distance))

        if self.model_ is None:
            if type(self.model_def) is str:
                input_size = (X.obsm['X_pca'].shape[-1],)
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

        hist = self.model_.fit(datagen,
                               epochs=self.epochs,
                               callbacks=[callback for callback in self.callbacks] + [EarlyStopping(monitor=loss_monitor,patience=self.n_epochs_without_progress)],
                               shuffle = shuffle_mode,
                               workers = 10,
                               verbose=self.verbose)

        self.loss_history_ += hist.history['loss']


    def fit(self, X, batch_name, Y=None, shuffle_mode=True):
        """Fit model.

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
        self._fit(X, batch_name, Y, shuffle_mode)
        return self

    def fit_transform(self, X, batch_name, Y=None, shuffle_mode=True):
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

        self.fit(X, batch_name, Y, shuffle_mode)
        return self.transform(X)

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.R0ckyyy123


        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.

        Returns
        -------
        X_new : array, shape (n_samples, embedding_dims)
            Embedding of the new data in low-dimensional space.
        """

        embedding = self.encoder.predict(X.obsm['X_pca'], verbose=self.verbose)
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

def nn_approx(ds1, ds2, names1 , names2, knn = 20, metric='euclidean', n_trees = 50):
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
          match.add((names1[a], names2[b_i]))

    return match

def calculate_mnns(adata, batch_name = "batch", knn = 20):

  batch_indices = []
  for i in adata.obs[batch_name].unique():
    batch_indices.append(np.where(adata.obs[batch_name] == i))

  matches = []
  for i in list(itertools.combinations(range(len(batch_indices)), 2)):
    print(i)
    ref = batch_indices[i[0]]
    new = batch_indices[i[1]]
    ds1 = adata[ref].obsm["X_pca"]
    ds2 = adata[new].obsm["X_pca"]
    names1 = adata[ref].obs_names
    names2 = adata[new].obs_names

    match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)
    match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)
    mutual = match1 & set([ (b, a) for a, b in match2 ])
    mutual = list(mutual)
    matches += mutual

  return(matches)

def calculate_knns(X, knn=20, n_trees = 10):
    a = AnnoyIndex(X.shape[1], metric="euclidean")
    for i in range(X.shape[0]):
        a.add_item(i, X[i, :])
    a.build(n_trees)

    ind = []
    for i in range(X.shape[0]):
        ind.append(a.get_nns_by_vector(X[i, :], knn, search_k=-1))
    neighbour_matrix = np.array(ind)

    return(neighbour_matrix)

def get_knn_pairs(neighbour_matrix, cell_names):
  matches = []
  for i in range(0, neighbour_matrix.shape[0]):
    neighbor_list = neighbour_matrix[i,:]
    anchor = cell_names[i]
    positive_pool = cell_names[neighbor_list]
    positive = np.random.choice(positive_pool)
    matches.append((anchor, positive))

  return(matches)

def find_leftover_cells(adata, matches):
  all_cells = adata.obs_names

  first = []
  second = []
  for i in matches:
    first.append(i[0])
    second.append(i[1])

  leftover = set(all_cells) - set(first).union(set(second))

  return(leftover)

def get_items(x, first, second, cell_names):
  names = second[np.where(first == x)].tolist()
  return [cell_names.get_loc(item) for item in names]

def convert_mnns_pairs_to_list(matches, cell_names):
  # Use parallelization to convert list of pairs into list of lists
  first = []
  second = []
  for i in matches:
    first.append(i[0])
    second.append(i[1])

  first = np.array(first)
  second = np.array(second)

  first_uniq = list(set(first))
  second_uniq = list(set(second) - set(first))

  if __name__ == '__main__':
      p = Pool(cpus)
      mnns_first = p.map(partial(get_items, first = first, second = second, cell_names = cell_names), first_uniq)
      mnns_second = p.map(partial(get_items, first = second, second = first, cell_names = cell_names), second_uniq)

  mnns_final = mnns_first + mnns_second

  return((mnns_final, first_uniq + second_uniq))

def convert_mnns_pairs_to_list_2(matches, cell_names):

  first = []
  second = []
  for i in matches:
    first.append(i[0])
    second.append(i[1])

  positives =[]
  anchors = []
  for ind in range(0, len(first)):
    first_tmp = first[ind]
    second_tmp = second[ind]

    if first_tmp not in anchors:
      anchors.append(first_tmp)
      positives.append([])

    if first_tmp in anchors:
      ind = anchors.index(first_tmp)
      positives[ind].append(cell_names.get_loc(second_tmp))

    if second_tmp not in anchors:
      anchors.append(second_tmp)
      positives.append([])

    if second_tmp in anchors:
      ind = anchors.index(second_tmp)
      positives[ind].append(cell_names.get_loc(first_tmp))

  return((positives, anchors))

def convert_mnns_pairs_to_list_3(matches, cell_names):

  G = nx.Graph()
  G.add_edges_from(matches)
  node_names = np.array(G.nodes)
  anchors = list(node_names)
  adj = nx.adjacency_matrix(G)

  tmp = np.split(adj.indices, adj.indptr[1:-1])

  positives = []
  for i in tmp:
    names = list(node_names[i])
    positives.append([cell_names.get_loc(x) for x in names])

  return((positives, anchors))
