from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import numpy as np
import argparse
import logging
import torch
import os

from src.utils import setup_device, setup_logging_level, plot_baseline_training_metrics, hungarian_accuracy
from src.training_procedures import train_baseline_model
from src.import_utils import import_dataset_with_name
from src.BaselineModel import BaselineModel


def argument_parser():
    """
    A parser to allow the user to easily experiment different parameters along with different datasets.
    """
    parser = argparse.ArgumentParser(usage='python Baseline.py [dataset_name] [options]',
                                     description='This program allows to run the baseline and compute the different performance metrics.')

    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['mnist', 'ForestCoverType', 'LetterRecognition', 'HumanActivityRecognition', 'Satimage', 'Pendigits', 'USCensus1990'],
                        help='The name of the dataset to import.')

    parser.add_argument('--use_cuda', type=str, default='True',
                        choices=['True', 'False'], required=False,
                        help='Set to True if you want the code to be run on your GPU. If set to False, code will run on CPU.')

    parser.add_argument('--log_lvl', type=str, default='info',
                        choices=['debug', 'info', 'warning'], required=False,
                        help='Change the log display level.')

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()

    device = setup_device(use_cuda=True if args.use_cuda == 'True' else False)

    setup_logging_level(args.log_lvl)

    # ====================== Step 1 - Import the datasets =======================
    logging.info("Importing " + args.dataset_name + "...")
    x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict, cat_columns_indexes = import_dataset_with_name(args.dataset_name, device)

    # We get the training and testing data for the known classes only
    x_test_known = x_test[np.in1d(y_test, y_train)]
    y_test_known = y_test[np.in1d(y_test, y_train)]

    x_test_unknown = x_test[np.in1d(y_test, y_unlab)]
    y_test_unknown = y_test[np.in1d(y_test, y_unlab)]

    # And then map them to {0, ..., |C|}
    mapper, ind = np.unique(y_train, return_inverse=True)
    mapping_dict = dict(zip(y_train, ind))

    # Map the training and testing values
    y_train_mapped = np.array(list(map(mapping_dict.get, y_train)))
    y_test_known_mapped = np.array(list(map(mapping_dict.get, y_test_known)))
    # ===========================================================================

    # =================== Step 2 - Define and train the model ===================
    model = BaselineModel(x_train.shape[1], len(np.unique(y_train)), p_dropout=0.2).to(device)

    logging.info(model)

    losses_dict = train_baseline_model(model, device, x_train, y_train_mapped, x_test_known, y_test_known_mapped, num_epochs=50)

    fig_path = os.path.join('.', 'figures', str(args.dataset_name) + '_baseline_training_metric.svg')
    logging.info('Saving metrics training curves in ' + str(fig_path))
    plot_baseline_training_metrics(losses_dict, fig_path)
    # ===========================================================================

    # ============ Step 3 - Train a K-Means on the penultimate layer ============
    model.eval()
    with torch.no_grad():
        x_unlab_projection = model.neural_net_forward(x_unlab)
        x_test_unknown_projection = model.neural_net_forward(x_test_unknown)
    model.train()

    km = KMeans(n_clusters=len(np.unique(y_unlab)))
    km_y_test_unknown_pred = km.fit_predict(x_test_unknown_projection.cpu())
    km_y_unlab_pred = km.fit_predict(x_unlab_projection.cpu())
    # ===========================================================================

    # ==================== Step 4 - Evaluate the prediction =====================
    train_acc, train_bacc = hungarian_accuracy(y_unlab, km_y_unlab_pred)
    train_ari = adjusted_rand_score(y_unlab, km_y_unlab_pred)
    train_nmi = normalized_mutual_info_score(y_unlab, km_y_unlab_pred)
    logging.info("Train clustering accuracy: {:.3f} / balanced acc.: {:.3f} / ari.: {:.3f} / nmi.: {:.3f}".format(train_acc, train_bacc, train_ari, train_nmi))

    test_acc, test_bacc = hungarian_accuracy(y_test_unknown, km_y_test_unknown_pred)
    test_ari = adjusted_rand_score(y_test_unknown, km_y_test_unknown_pred)
    test_nmi = normalized_mutual_info_score(y_test_unknown, km_y_test_unknown_pred)
    logging.info("Test clustering accuracy: {:.3f} / balanced acc.: {:.3f} / ari.: {:.3f} / nmi.: {:.3f}".format(test_acc, test_bacc, test_ari, test_nmi))
    # ===========================================================================
