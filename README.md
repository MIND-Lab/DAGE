# Deep Attributed Graph Embedding
This the official repository of DAGE (Deep Attributed Graph Embedding), where both the source code and the datasets are available.
DAGE has been published at MDAI 2022, [click here to read the paper](https://link.springer.com/chapter/10.1007/978-3-031-13448-7_15). 

If you decide to use this resource, please cite:

::

@InProceedings{10.1007/978-3-031-13448-7_15,
author="Fersini, Elisabetta
and Mottadelli, Simone
and Carbonera, Michele
and Messina, Enza",
editor="Torra, Vicen{\c{c}}
and Narukawa, Yasuo",
title="Deep Attributed Graph Embeddings",
booktitle="Modeling Decisions for Artificial Intelligence",
year="2022",
publisher="Springer International Publishing",
pages="181--192",
}

# Requirements
- python 3.8.8
- tensorflow-gpu 2.3.0 
- scikit-learn 0.24.1 
- keras 2.4.3 
- networkx 2.5.1 
- tqdm 4.59.0 

# How to run
To run DAGE on either DBLP or Citeseer-M10, please follow these steps:
- make sure all the requirements above are satisfied;
- open the config.py file,
- edit the dataset_name variable with either "dblp" or "citeseer-m10";
- configure the hyper-parameters of DAGE or leave them as they are;
- close the config.py file;
- execute main.py (i.e., python main.py)
