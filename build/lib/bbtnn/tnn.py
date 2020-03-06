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


def generator_from_index(adata, k = 20, batch_size = 32, search_k=-1,
                         precompute=True, verbose=1):

    batch_list = adata.obs['batch']
    datasets = []
    for i in batch_list.unique():
          datasets.append(adata[batch_list == i])

    datasets_pcs = []
    for i in batch_list.unique():
          datasets_pcs.append(adata[batch_list == i].obsm["X_pca"])

    alignments, matches = find_alignments(datasets = datasets_pcs, knn = k, prenormalized = True, approx = True)

    dict_mnn, cells_for_mnn = create_dictionary_mnn(datasets, matches)

    cells_for_knn = list(set(adata.obs_names) - set(cells_for_mnn))

    dict_knn, cells_for_knn_1 = create_dictionary_knn(adata, cells_for_knn, k)

    print ('******Batches:'+ str(batch_list.unique()))
    print ('******Total number of cells:'+ str(len(adata.obs_names)))
    print ('******Number of cells for MNN:'+ str(len(cells_for_mnn)))

    knn_frame = pd.DataFrame({'index':list(dict_knn.keys()), 'neighbor':list(dict_knn.values())})
    mnn_frame = pd.DataFrame({'index':list(dict_mnn.keys()), 'neighbor':list(dict_mnn.values())})
    merged_frame = mnn_frame.append(knn_frame)

    triplet_list = []
    for i in adata.obs_names:
        if i in dict_mnn.keys():
            samples = dict_mnn[i]
            samples_indices = [adata.obs_names.get_loc(x) for x in samples]
            triplet_list.append(samples_indices)
        else:
            if i in dict_knn.keys():
                samples = dict_knn[i]
                samples_indices = [adata.obs_names.get_loc(x) for x in samples]
                triplet_list.append(samples_indices)

    return KnnTripletGenerator(X = adata.obsm["X_pca"], dictionary = triplet_list, batch_size=batch_size)


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

        anchor = row_index
        positive = np.random.choice(neighbour_list)
        negative = np.random.randint(self.num_cells)

        triplets += [self.X[anchor], self.X[positive],
                     self.X[negative]]

        return triplets

def merge_dict(x,y):
    for k,v in x.items():
                if k in y.keys():
                    y[k] += v
                else:
                    y[k] = v
    return y

def create_dictionary_mnn(datasets, matches):
    pairs = []
    for i in matches.keys():
        first = datasets[i[0]].obs_names.tolist()
        second = datasets[i[1]].obs_names.tolist()
        for j in matches[i]:
            pairs.append([first[j[0]], second[j[1]]])

    unique = np.unique(np.array(pairs)[:,0])
    pairs_one = np.array(pairs)[:,0]
    pairs_two = np.array(pairs)[:,1]
    data = pd.DataFrame({"pair_one" : pairs_one, "pair_two" : pairs_two})
    data_dict1 = data.groupby("pair_one").pair_two.apply(list).to_dict()
    data_dict2 = data.groupby("pair_two").pair_one.apply(list).to_dict()

    cell_for_mnn = set(data['pair_one'].unique()) | set(data['pair_two'].unique())
    #dict_mnn = {**data_dict1, **data_dict2} ##This will overwrite the values
    dict_mnn = merge_dict(data_dict1, data_dict2)

    return(dict_mnn, cell_for_mnn)

def create_dictionary_knn(adata, cells_for_knn, k):

    dataset = adata[cells_for_knn]
    batch_list = dataset.obs['batch']
    pairs=[]
    for i in batch_list.unique():
        pp=[]
        dataset_ref = dataset[batch_list==i]
        dataset_ref_pcs = dataset_ref.obsm['X_pca']
        dataset_new = adata[adata.obs['batch']==i]
        dataset_new_pcs = dataset_new.obsm['X_pca']

        match_self = nn_approx(dataset_ref_pcs, dataset_new_pcs,  knn = k)
        names_knn = dataset_ref.obs_names.tolist()
        names_all = dataset_new.obs_names.tolist()
        for j in match_self:
            pairs.append([names_knn[j[0]], names_all[j[1]]])

    pairs_one = np.array(pairs)[:,0]
    pairs_two = np.array(pairs)[:,1]
    data = pd.DataFrame({"pair_one" : pairs_one, "pair_two" : pairs_two})
    dict_knn = data.groupby("pair_one").pair_two.apply(list).to_dict()

    cell_for_knn = set(data['pair_one'].unique())

    print (len(cell_for_knn))

    return(dict_knn, cell_for_knn)

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

    def _fit(self, X, Y=None, shuffle_mode=True):

        datagen = generator_from_index(X,
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


    def fit(self, X, Y=None, shuffle_mode=True):
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

ALPHA = 0.10
APPROX = True
BATCH_SIZE = 5000
DIMRED = 100
KNN = 20
N_ITER = 500
SIGMA = 15
VERBOSE = 2

def find_alignments(datasets, knn=KNN, approx=APPROX, verbose=VERBOSE,
                    alpha=ALPHA, prenormalized=False,):
    table1, _, matches = find_alignments_table(
        datasets, knn=knn, approx=approx, verbose=verbose,
        prenormalized=prenormalized,
    )

    alignments = [ (i, j) for (i, j), val in reversed(
        sorted(table1.items(), key=operator.itemgetter(1))
    ) if val > alpha ]

    return alignments, matches

def find_alignments_table(datasets, knn=KNN, approx=APPROX, verbose=VERBOSE,
                          prenormalized=False):
    if not prenormalized:
        datasets = [ normalize(ds, axis=1) for ds in datasets ]

    table = {}
    for i in range(len(datasets)):
        if len(datasets[:i]) > 0:
            fill_table(table, i, datasets[i], datasets[:i], knn=knn,
                       approx=approx)
        if len(datasets[i+1:]) > 0:
            fill_table(table, i, datasets[i], datasets[i+1:],
                       knn=knn, base_ds=i+1, approx=approx)
    # Count all mutual nearest neighbors between datasets.
    matches = {}
    table1 = {}
    if verbose > 1:
        table_print = np.zeros((len(datasets), len(datasets)))
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if i >= j:
                continue
            if not (i, j) in table or not (j, i) in table:
                continue
            match_ij = table[(i, j)]
            match_ji = set([ (b, a) for a, b in table[(j, i)] ])
            matches[(i, j)] = match_ij & match_ji

            table1[(i, j)] = (max(
                float(len(set([ idx for idx, _ in matches[(i, j)] ]))) /
                datasets[i].shape[0],
                float(len(set([ idx for _, idx in matches[(i, j)] ]))) /
                datasets[j].shape[0]
            ))
            if verbose > 1:
                table_print[i, j] += table1[(i, j)]

    if verbose > 1:
        print(table_print)
        return table1, table_print, matches
    else:
        return table1, None, matches

def fill_table(table, i, curr_ds, datasets, base_ds=0,
               knn=KNN, approx=APPROX):
    curr_ref = np.concatenate(datasets)
    if approx:
        match = nn_approx(curr_ds, curr_ref, knn=knn)
    else:
        match = nn(curr_ds, curr_ref, knn=knn, metric_p=1)

    # Build interval tree.
    itree_ds_idx = IntervalTree()
    itree_pos_base = IntervalTree()
    pos = 0
    for j in range(len(datasets)):
        n_cells = datasets[j].shape[0]
        itree_ds_idx[pos:(pos + n_cells)] = base_ds + j
        itree_pos_base[pos:(pos + n_cells)] = pos
        pos += n_cells

    # Store all mutual nearest neighbors between datasets.
    for d, r in match:
        interval = itree_ds_idx[r]
        assert(len(interval) == 1)
        j = interval.pop().data
        interval = itree_pos_base[r]
        assert(len(interval) == 1)
        base = interval.pop().data
        if not (i, j) in table:
            table[(i, j)] = set()
        table[(i, j)].add((d, r - base))
        assert(r - base >= 0)


# Find mutual nearest neighbors.
def mnn(ds1, ds2, knn=KNN, approx=APPROX):
    # Find nearest neighbors in first direction.
    if approx:
        match1 = nn_approx(ds1, ds2, knn=knn)  #should be a list
    else:
        match1 = nn(ds1, ds2, knn=knn)

    # Find nearest neighbors in second direction.
    if approx:
        match2 = nn_approx(ds2, ds1, knn=knn)
    else:
        match2 = nn(ds2, ds1, knn=knn)

    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual

# Exact nearest neighbors search.
def nn(ds1, ds2, knn=KNN, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

# Approximate nearest neighbors using locality sensitive hashing.
def nn_approx(ds1, ds2, knn = KNN, metric='euclidean', n_trees = 50):
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
            match.add((a, b_i))

    return match
