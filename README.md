![bbtnn](https://github.com/lkmklsmn/bbtnn/blob/master/examples/test1.png)

# bbtnn
**b**atch **b**alanced **t**riplet **n**eural **n**etwork

Environment:Python

bbtnn's latest version: http://github.com/lkmklsmn/bbtnn....


# Usage
The bbtnn package is implementation of batch correction and deep embedding for scRNA-seq. With bbtnn, you can:

* Integrate scRNA-seq datasets across batches with/without labels.
* Build a low-dimensional representation of the scRNA-seq data.
* Obtain soft-clustering assignments of cells.

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
pip install

```

### Anaconda

If you do not have Python3.x.x installed, consider installing Anaconda. After installing Anaconda, you can create a new environment, for example, BBTNN (you can change to any name you like):

```alias
conda create -n BBTNN python=3.6.3
~activate your environment 
source activate BBTNN 
git clone http://github.com/lkmklsmn/bbtnn
cd bbtnn
python setup.py build
python setup.py install
~ now you can check whether `bbtnn` installed successfully!
```

# Examples

```
from bbtnn import bbtnn as bt
embedding=bt.unsupervised_bbtnn(X= adata.obsm["X_pca"], Y = None, batch = adata.obs["batch"], verbose = 0, model = "szubert",                                   n_pcs=30)

adata.obsm["X_umap"]=embedding
sc.pl.umap(adata, color = ["batch", "celltype"])
```

# Contributing
Source code: Github
We are continuing adding new features. Bug reports or feature requests are welcome.

# References
Please consider citing the following reference:

