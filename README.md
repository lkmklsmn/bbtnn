![bbtnn](https://github.com/lkmklsmn/bbtnn/blob/master/examples/test1.png)

# bbtnn
**b**atch **b**alanced **t**riplet **n**eural **n**etwork

The bbtnn calculates a deep integrated embedding for scRNA-seq data. With bbtnn, you can:

* Integrate scRNA-seq datasets across batches with/without labels.
* Build a low-dimensional representation of the scRNA-seq data.


# Check out out live tutorial!
The following notebooks allow you to interactively explore BBTNN and can be run within your browser. We have prepared two analysis examples:
1. [Simulation dataset](https://colab.research.google.com/github/lkmklsmn/...)
2. [Pancreas dataset](https://github.com/lkmklsmn/bbtnn/tree/master/examples/TNN_pancreas_comparison.ipynb)

# Installation

To install **bbtnn**, you must make sure that your python version is 3.x.x. 

Now you can install the current release of bbtnn by the following ways:

### Pypi 

Directly install the package from Pypi.

```alias
$ pip install bbtnn
```

### Github

Download the package from Github and install it locally:

```alias
git clone http://github.com/lkmklsmn/bbtnn
cd bbtnn
pip install .
```
# Input
## Unsupervised model
1. Adata with PCs
2. Batch name

## Label supervised model (Triplet generated based on the known labels)
1. Adata with PCs
2. Batch name
3. Celltype name

# Result
1. Coordinates for the embedding layer 
2. Visualization of the embedding layer


# Usage
## Unsupervised model

```alias
from bbtnn.tnn_v3 import BBTNN
model = BBTNN(k=50, distance='pn', batch_size=32, n_epochs_without_progress=10, k_to_m_ratio = 0.01)
model.fit(X = adata, Y = None, batch_name='batch')
```

## Supervised model
```alias
model = BBTNN(k=50, distance='pn', batch_size=64, n_epochs_without_progress=10, approx = False)
model.fit(X = adata, Y = None, batch_name='batch', celltype_name='Celltypes', cell_labeled = True)
```
## Semi-supervised model

```alias
model = BBTNN(k=50, distance='pn', batch_size=64, n_epochs_without_progress=10, approx = False)
model.fit(X = adata, Y = None, batch_name='batch', celltype_name='Celltypes', cell_labeled = True, mask_batch= batch_name)
```
