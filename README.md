# Deep Attributed Graph Embedding
This repository contains both the source code of DAGE and the datasets used during the evaluation.

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

# Pubblication
Fersini, E. and Messina, E. and Mottadelli, S. P. (2021). Deep Attributed Graph Embedding. In Proceedings of the 29th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning. (SUBMITTED)
