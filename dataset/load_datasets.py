import json
import networkx as nx
import numpy as np


def loadGraphFromEdgeListTxt(file_name, is_directed_graph=False):
    """
    This function loads a graph from file and returns it in nx format

    :param file_name: the file containing the list of edges
    :param is_directed_graph: whether the graph is directed or undirected
    :return: a graph in nx format
    """

    with open(file_name, 'r') as f:
        if is_directed_graph:
            g = nx.DiGraph()
        else:
            g = nx.Graph()
        for line in f:
            edge = line.strip().split()
            if len(edge) == 3:
                w = float(edge[2])
            else:
                w = 1.0
            if len(edge) == 1:
                g.add_node(int(edge[0]))
            else:
                g.add_edge(int(edge[0]), int(edge[1]), weight=w)
    return g


def get_adjacency_matrix(edge_list_file):
    """
    This function returns the adjacency matrix of the graph given the list of edges

    :param edge_list_file: file containing the list of edges of the graph
    :return: the adjacency matrix W of the graph
    """

    # load the graph in networkx format
    g = loadGraphFromEdgeListTxt(edge_list_file)

    # get the adjacency matrix
    W = nx.to_scipy_sparse_matrix(g, nodelist=sorted(g.nodes()))
    return W


def get_labels(label_file):
    """
    This function returns a vector containing the labels of the nodes

    :param label_file: the file containing the labels of the ndoes
    :return: the label vector
    """

    # get the dictionary {node_id : node_label, ...}
    label_file = open(label_file, "r")
    node_label_dict = {}
    for line in label_file:
        split_line = line.split(" ")
        node_id = int(split_line[0])
        node_label = int(split_line[1].replace("\n", ""))
        node_label_dict[node_id] = node_label

    # sort the keys (node_ids) of the dictionary
    node_label_dict = {k: node_label_dict[k] for k in sorted(node_label_dict)}
    labels = np.array(list(node_label_dict.values()))
    return labels



def doc2vec(doc, dim):
    """
    This function returns a vector corresponding to the input document
    :param doc: a dictionary of the form {word_id : freq}
    :param dim: number of words in the dictionary
    :return: a vector representing the document
    """

    vec = np.zeros(dim)
    for idx, val in doc.items():
        vec[int(idx)] = val
    return vec


def vecnorm(vec, norm, epsilon=1e-3):
    """
    This function scales a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.

    :param vec: vector to be scaled
    :param norm: normalization method
    :param epsilon: a small number
    :return: a normalized vector
    """

    if norm not in ('prob', 'max1', 'logmax1'):
        raise ValueError("'%s' is not a supported norm. Currently supported norms include 'prob',\
             'max1' and 'logmax1'." % norm)

    if isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=float)
        veclen = None
        if norm == 'prob':
            veclen = np.sum(np.abs(vec)) + epsilon * len(vec) # smoothing
        elif norm == 'max1':
            veclen = np.max(vec) + epsilon
        elif norm == 'logmax1':
            vec = np.log10(1. + vec)
            veclen = np.max(vec) + epsilon

        if veclen > 0.0:
            return (vec + epsilon) / veclen
        else:
            return vec
    else:
        raise ValueError('vec should be ndarray, found: %s' % type(vec))


def get_attribute_matrix(corpus_path):
    """
    This function generates and returns the attribute matrix A such that A[i, :] is the attribute vector associated to
    the i-th vertex of the graph

    :param corpus_path: the path to the file containing the corpus structured in json format
    :return: the attribute matrix A
    """

    # load the corpus from corpus_path
    try:
        with open(corpus_path, 'r') as datafile:
            corpus = json.load(datafile)
    except Exception as e:
        raise e

    n_vocab = len(corpus['vocab'])

    # sort the corpus dictionary by id
    docs = corpus['docs']
    docs = {int(k): v for k, v in docs.items()}
    docs = {k : docs[k] for k in sorted(docs)}
    corpus.clear()  # save memory

    # create the attribute matrix
    A = []
    for k in list(docs):
        A.append(vecnorm(doc2vec(docs[k], n_vocab), 'logmax1', 0))
    A = np.r_[A]

    return A


def load_dataset(dataset_folder_path):
    """
    This function loads the adjacency matrix, the attribute matrix and the class label vector from the dataset files

    :param dataset_folder_path: the path to the folder containing the dataset
    :return: the adjacency matrix W, the attribute matrix A, the class label vector L
    """

    edge_list_file_path = dataset_folder_path + "/edgelist.txt"
    attribute_file_path = dataset_folder_path + "/processed_features.txt"
    labels_file_path = dataset_folder_path + "/labels.txt"
    A = get_attribute_matrix(attribute_file_path)
    W = get_adjacency_matrix(edge_list_file_path)
    L = get_labels(labels_file_path)

    return W, A, L
