from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import argparse
import logging
import torch
import math

from src.import_utils import import_dataset_with_name
from src.utils import setup_logging_level, hungarian_accuracy, setup_device


def argument_parser():
    """
    A parser to allow the user to easily experiment different parameters along with different datasets.
    """
    parser = argparse.ArgumentParser(usage='python Clustering.py [dataset_name] [options]',
                                     description='This program allows to run a K-Means or a Spectral Clustering and compute the different performance metrics.')

    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['mnist', 'ForestCoverType', 'LetterRecognition', 'HumanActivityRecognition', 'Satimage', 'Pendigits', 'USCensus1990'],
                        help='The name of the dataset to import.')

    parser.add_argument('--method', type=str, required=True,
                        choices=['kmeans', 'default_sc', 'cosine_sc'],
                        help='The clustering method to use. \'sc\' refers to Spectral Clustering, and can be used with either the default implementation or the cosine similarity matrix.')

    parser.add_argument('--log_lvl', type=str, default='info',
                        choices=['debug', 'info', 'warning'], required=False,
                        help='Change the log display level.')

    parser.add_argument('--train_or_test', type=str, default='test', required=False,
                        choices=['train', 'test'],
                        help='Compute on either the train or the test split.')

    parser.add_argument('--k_clusters', type=int, default=None, required=False,
                        help='The number of clusters to used. Default is the ground truth.')

    parser.add_argument('--n_runs', type=int, default=1, required=False,
                        help='The number of executions, the results will be averaged.')

    return parser.parse_args()


def normalize_x(input_x):
    """
    Simple method to normalize the input_x to [0, 1].
    :param input_x: torch.tensor: The tensor to normalize.
    :return:
        torch.tensor: The normalized tensor.
    """
    input_x -= input_x.min(1, keepdim=True)[0]
    input_x /= input_x.max(1, keepdim=True)[0]
    return input_x


def batch_pairwise_cosine_similarity(input_x, batch_size=5):
    """
    Computation of the pairwise cosine similarity matrix split per batch to avoid memory errors.
    :param input_x: torch.tensor: The batch to compute the similarity matrix on, shape (n_samples, n_features).
    :param batch_size: int: The batch size used when computing the similarity matrix. Reduce if you have memory errors.
    :return:
        torch.tensor: The pairwise cosine similarity matrix of shape (n_samples, n_features).
    """
    sim_matrix = torch.zeros((len(input_x), len(input_x)))
    n_batchs = math.ceil((input_x.shape[0]) / batch_size)
    batch_start_index, batch_end_index = 0, min(batch_size, len(input_x))
    for _ in tqdm(range(n_batchs)):
        batch_length = batch_end_index - batch_start_index
        sim_matrix[:, batch_start_index:batch_end_index] = nn.CosineSimilarity(dim=1)(input_x.repeat_interleave(batch_length, dim=0), input_x[batch_start_index:batch_end_index].repeat(len(input_x), 1)).reshape(len(input_x), -1)
        batch_start_index += batch_size
        batch_end_index = min((batch_end_index + batch_size), input_x.shape[0])

    return sim_matrix


if __name__ == '__main__':
    args = argument_parser()

    device = setup_device(use_cuda=False)

    setup_logging_level(args.log_lvl)

    # ====================== Step 1 - Import the datasets =======================
    logging.info("Importing " + args.dataset_name + "...")
    x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict, cat_columns_indexes = import_dataset_with_name(args.dataset_name, device)

    if args.train_or_test == 'train':
        x = x_unlab
        y = y_unlab
    elif args.train_or_test == 'test':
        x = x_test[np.in1d(y_test, np.unique(y_unlab))]
        y = y_test[np.in1d(y_test, np.unique(y_unlab))]
    else:
        raise ValueError('Undefined value for parameter --train_or_test.')
    # ===========================================================================

    # =================== Step 2 - Train the clustering model ===================
    k_clusters = args.k_clusters
    if k_clusters is None:
        k_clusters = len(np.unique(y_unlab))
    logging.info('Using k_clusters = ' + str(k_clusters))

    acc_list, bacc_list = [], []
    nmi_list, ari_list = [], []

    if args.method == 'cosine_sc':
        normalized_x = normalize_x(x)  # We normalize the input to only get positive values in the pairwise cosine similarity matrix
        logging.info('Computing the pairwise cosine similarity matrix...')
        x_similarity_matrix = batch_pairwise_cosine_similarity(normalized_x).cpu().numpy()

    logging.info('Training the clustering model' + ('s' if args.n_runs > 1 else '') + '...')
    for _ in tqdm(range(args.n_runs)):
        if args.method == 'kmeans':
            clustering_model = KMeans(n_clusters=k_clusters)
            clustering_pred = clustering_model.fit_predict(x.cpu())
        elif args.method == 'default_sc':
            clustering_model = SpectralClustering(n_clusters=k_clusters, affinity='nearest_neighbors')
            clustering_pred = clustering_model.fit_predict(x.cpu())
        elif args.method == 'cosine_sc':
            clustering_model = SpectralClustering(n_clusters=k_clusters, affinity='precomputed')
            clustering_pred = clustering_model.fit_predict(x_similarity_matrix)
        else:
            raise ValueError('Undefined value for parameter --method.')

        acc, bacc = hungarian_accuracy(y, clustering_pred)
        ari = adjusted_rand_score(y, clustering_pred)
        nmi = normalized_mutual_info_score(y, clustering_pred)

        acc_list.append(acc)
        bacc_list.append(bacc)
        nmi_list.append(nmi)
        ari_list.append(ari)
    # ===========================================================================

    logging.info(args.train_or_test + " clustering accuracy: {:.3f}±{:.3f} / balanced acc.: {:.3f}±{:.3f} / ari.: {:.3f}±{:.3f} / nmi.: {:.3f}±{:.3f}".format(np.array(acc_list).mean(), np.array(acc_list).std(), np.array(bacc_list).mean(), np.array(bacc_list).std(), np.array(ari_list).mean(), np.array(ari_list).std(), np.array(nmi_list).mean(), np.array(nmi_list).std()))
