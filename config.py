########################
##   INPUT SETTINGS   ##
########################
dataset_name = "citeseer-m10"           # name of the dataset
d = 128                                 # embedding dimension
alpha = 1                               # hyper-parameter for the second-order proximity loss term
beta = 1                                # hyper-parameter for the semantic proximity loss term
gamma = 1                               # hyper-parameter for the consistency loss term
beta_w = 5                              # hyper-parameter for the adjacency matrix reconstruction
beta_a = 5                              # hyper-parameter for the attribute matrix reconstruction
epochs = 2000                              # number of epochs
lr = 1e-4                               # learning rate
batch_size = 128                        # batch size


#########################################################
##   output settings (WARNING: should not be edited)   ##
#########################################################
root_path = "./"
path_result = root_path + "experiments/results/" + dataset_name + "/node_classification_results.txt"
path_embedding = root_path + "embedding/" + dataset_name + "/embedding_" + dataset_name + "_" + str(d) + ".txt"
path_embedding_id = root_path + "embedding/" + dataset_name + "/embedding_id_" + dataset_name + "_" + str(d) + ".txt"