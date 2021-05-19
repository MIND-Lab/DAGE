import config
import os
import numpy as np
import functools as ft
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, rand_score
from tqdm import tqdm
from sklearn import model_selection as sk_ms
from sklearn.svm import LinearSVC
from operator import methodcaller

def evaluate_node_classification_repeat(X, Y):
    """
    This function repeats 10 times the node classification evaluation.
    It stores the results in a file.

    :param X: the node embeddings
    :param Y: the node labels
    """

    repeat = 10
    test_split_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    round_result_filepath = config.path_result
    perf_measure_str = ["accuracy",
                        "precision_macro",
                        "precision_micro",
                        "recall_macro",
                        "recall_micro",
                        "f1_macro",
                        "f1_micro"]

    perf_measure_funct = [accuracy_score,
                          ft.partial(precision_score, average='macro'),
                          ft.partial(precision_score, average='micro'),
                          ft.partial(recall_score, average='macro'),
                          ft.partial(recall_score, average='micro'),
                          ft.partial(f1_score, average='macro'),
                          ft.partial(f1_score, average='micro')
                          ]

    if not os.path.isdir(config.path_result[:config.path_result.rfind("/") + 1]):
        os.mkdir(config.path_result[:config.path_result.rfind("/") + 1])

    if os.path.isfile(round_result_filepath):
        print(
            "[INFO] File \"" + round_result_filepath + "\" already exists. Please, move it to another folder if you want to"
                                                       " perform the experiments")
        return

    round_result_file = open(round_result_filepath, 'w')
    perf = {key: [None] * repeat for key in perf_measure_str}
    for round_id in tqdm(range(repeat)):  # for each round
        perf_round = {key: [None] * len(test_split_thresholds) for key in perf_measure_str}
        for i, test_split_threshold in enumerate(test_split_thresholds):  # for each test split threshold
            perf_measure_current_split = evaluate_node_classification_single(
                X,
                Y,
                test_split_threshold,
                perf_measure_funct=perf_measure_funct
            )

            for j, perf_measure_name in enumerate(perf_measure_str):
                perf_round[perf_measure_name][i] = perf_measure_current_split[j]

        for j, perf_measure_name in enumerate(perf_measure_str):
            perf[perf_measure_name][round_id] = perf_round[perf_measure_name]

    for perf_measure_name in (perf_measure_str):
        round_result_file.write(
            '%s\t%s\n' % (perf_measure_name, '\t'.join(map(str, np.mean(perf[perf_measure_name], 0)))))
        round_result_file.write(
            '%s (std)\t%s\n' % (perf_measure_name, '\t'.join(map(str, np.std(perf[perf_measure_name], 0)))))
    round_result_file.close()

def evaluate_node_classification_single(X, Y, test_split_threshold, perf_measure_funct=None):
    """
    This function performs the node classification evaluation.

    :param X: the node embeddings
    :param Y: the node labels
    :param type_classifier: name of the classifier
    :param test_split_threshold: the vector containing the teste split thresholds
    :param perf_measure_funct: list of performance measures to compute
    :return: the results of the computed performance measures
    """
    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, test_size=test_split_threshold)

    classifier = LinearSVC()

    classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_test)
    perf_measure_results = list(map(methodcaller('__call__', Y_test, predictions), perf_measure_funct))
    return perf_measure_results
