import numpy as np
import argparse
import logging
import json
import os

from src.utils import setup_device, plot_alternative_joint_learning_metrics, setup_logging_level
from src.training_procedures import joint_training, vime_training
from src.import_utils import import_dataset_with_name
from src.TabularNCDModel import TabularNCDModel


def restricted_float(x):
    """
    Check function for the argument_parser.
    The passed value must be a float between 0.0 and 1.0.
    :param x: value to check.
    :return: x.
    """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def argument_parser():
    """
    A parser to allow the user to easily experiment different parameters along with different datasets.
    """
    parser = argparse.ArgumentParser(usage='python TabularNCD.py [dataset_name] [options]',
                                     description='This program allows to run the two training steps of TabularNCD and compute the different performance metrics.')

    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['mnist', 'ForestCoverType', 'LetterRecognition', 'HumanActivityRecognition', 'Satimage', 'Pendigits', 'USCensus1990'],
                        help='The name of the dataset to import.')

    parser.add_argument('--use_cuda', type=str, default='True',
                        choices=['True', 'False'], required=False,
                        help='Set to True if you want the code to be run on your GPU. If set to False, code will run on CPU.')

    parser.add_argument('--log_lvl', type=str, default='info',
                        choices=['debug', 'info', 'warning'], required=False,
                        help='Change the log display level.')

    parser.add_argument('--hyper_parameters_path', type=str, default=None, required=False,
                        help='Path to the hyper-params file. Set to \'auto\' to find it in .\\data\\dataset_name\\hyperparameters.json')

    parser.add_argument('--ssl_lr', type=restricted_float, default=0.001, required=False,
                        help='Learning rate of the self-supervised learning phase.')

    parser.add_argument('--lr_classif', type=float, default=0.001, required=False,
                        help='Learning rate of the classification network in the joint learning phase.')

    parser.add_argument('--lr_cluster', type=float, default=0.001, required=False,
                        help='Learning rate of the clustering network in the joint learning phase.')

    parser.add_argument('--encoder_layers_sizes', nargs='+', type=int, default=None, required=False,
                        help=' The sizes of the encoder\'s layers. Must include the input and output sizes.')

    parser.add_argument('--ssl_layers_sizes', nargs='+', type=int, default=[], required=False,
                        help='The hidden layers sizes of the mask and feature vector estimators. Do not include input and output sizes.')

    parser.add_argument('--joint_learning_layers_sizes', nargs='+', type=int, default=[], required=False,
                        help='The hidden layers sizes of the classification and clustering networks. Do not include input and output sizes.')

    parser.add_argument('--activation_fct', type=str, default='relu', required=False,
                        choices=['relu', 'sigmoid', 'tanh', None], help='The activation function used in the hidden layers of the encoder.')

    parser.add_argument('--encoder_last_activation_fct', type=str, default=None, required=False,
                        choices=['relu', 'sigmoid', 'tanh', None], help='The activation function of the very last layer of the encoder.')

    parser.add_argument('--ssl_last_activation_fct', type=str, default=None, required=False,
                        choices=['relu', 'sigmoid', 'tanh', None], help='The activation function of the very last layer of the feature estimator network.')

    parser.add_argument('--joint_last_activation_fct', type=str, default=None, required=False,
                        choices=['relu', 'sigmoid', 'tanh', None], help='The activation function of the very last layer of the classification and clustering networks.')

    parser.add_argument('--dropout', type=restricted_float, default=0.0, required=False,
                        help='The dropout probability')

    parser.add_argument('--p_m', type=restricted_float, default=0.3, required=False,
                        help='Corruption probability')

    parser.add_argument('--alpha', type=float, default=2.0, required=False,
                        help='Loss_vime = mask_estim_loss + alpha * feature_estim_loss')

    parser.add_argument('--batch_size', type=int, default=512, required=False,
                        help='Batch size of the joint learning step.')

    parser.add_argument('--cosine_topk', type=restricted_float, default=None, required=False,
                        help='The percentage of the maximum number of pairs in a mini-batch that are considered positive.')

    parser.add_argument('--M', type=int, default=2000, required=False,
                        help='Size of the memory queue for the data augmentation method.')

    parser.add_argument('--epochs', type=int, default=30, required=False,
                        help='Number of joint training epochs.')

    parser.add_argument('--transform_method', type=str, default='new_2', required=False,
                        choices=['old', 'old_2', 'new_1', 'new_2'], help='The variation of the SMOTE-NC insipired method. The \'old\' versions cannot handle categorical features.')

    parser.add_argument('--k_neighbors', type=int, default=1, required=False,
                        help='The number of neighbors to consider in the data augmentation method.')

    parser.add_argument('--w1', type=restricted_float, default=0.8, required=False,
                        help='The classification network trade-off parameter.')

    parser.add_argument('--w2', type=restricted_float, default=0.8, required=False,
                        help='The clustering network trade-off parameter.')

    parser.add_argument('--pseudo_labels_method', type=str, default='top_k_cosine_per_instance', required=False,
                        choices=['cosine', 'top_k_cosine', 'top_k_cosine_faster', 'top_k_cosine_per_instance', 'ranking'], help='The pseudo labels definition method.')

    parser.add_argument('--use_ssl', type=str, default='True',
                        choices=['True', 'False'], help='Use SSL to initialize the encoder or not.')

    parser.add_argument('--freeze_weights', type=str, default='True',
                        choices=['True', 'False'], help='Freeze the weights of the encoder\'s layer (except the last one) after SSL initialization or not.')

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()

    device = setup_device(use_cuda=True if args.use_cuda == 'True' else False)

    setup_logging_level(args.log_lvl)

    logging.info("Importing " + args.dataset_name + "...")
    x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict, cat_columns_indexes = import_dataset_with_name(args.dataset_name, device)

    # Read the hyper-parameters file:
    hyper_parameters_path = args.hyper_parameters_path
    if hyper_parameters_path == 'auto':
        config = json.load(open(os.path.join('.', 'data', args.dataset_name, 'hyperparameters.json')))
    elif hyper_parameters_path is not None:
        config = json.load(open(hyper_parameters_path))
    else:
        logging.info("No hyper-parameter file defined, using passed hyper-parameter values.")

        if args.cosine_topk is None:
            logging.warning('No value defined for --cosine_topk, using sub-optimal value 10 (%). Please define a value.')

        config = {
            'encoder_layers_sizes': args.encoder_layers_sizes if args.encoder_layers_sizes is not None else [x_train.shape[1], x_train.shape[1], x_train.shape[1]],
            'cosine_topk': args.cosine_topk if args.cosine_topk is not None else 10.0,
            'ssl_layers_sizes': args.ssl_layers_sizes,
            'joint_learning_layers_sizes': args.joint_learning_layers_sizes, 'activation_fct': args.activation_fct,
            'encoder_last_activation_fct': args.encoder_last_activation_fct, 'ssl_last_activation_fct': args.ssl_last_activation_fct,
            'joint_last_activation_fct': args.joint_last_activation_fct, 'dropout': args.dropout, 'p_m': args.p_m,
            'alpha': args.alpha, 'batch_size': args.batch_size, 'M': args.M,
            'epochs': args.epochs, 'transform_method': args.transform_method, 'k_neighbors': args.k_neighbors,
            'w1': args.w1, 'w2': args.w2, 'pseudo_labels_method': args.pseudo_labels_method, 'use_ssl': args.use_ssl,
            'freeze_weights': args.freeze_weights, 'lr_classif': args.lr_classif, 'lr_cluster': args.lr_cluster
        }

    logging.debug('The current hyper-parameters are:')
    logging.debug(config)

    # Declare the model
    model = TabularNCDModel(encoder_layers_sizes=config['encoder_layers_sizes'],
                            ssl_layers_sizes=config['ssl_layers_sizes'],
                            joint_learning_layers_sizes=config['joint_learning_layers_sizes'],
                            n_known_classes=len(np.unique(y_train)) + 1,
                            n_unknown_classes=len(np.unique(y_unlab)),
                            activation_fct=config['activation_fct'],
                            encoder_last_activation_fct=config['encoder_last_activation_fct'],
                            ssl_last_activation_fct=config['ssl_last_activation_fct'],
                            joint_last_activation_fct=config['joint_last_activation_fct'],
                            p_dropout=config['dropout']).to(device)
    logging.info(model)

    # ==================== Step 1 - Self-Supervised Learning ====================
    if config['use_ssl'] == 'True' or config['use_ssl'] is True:
        logging.info('Starting self-supervised learning...')
        vime_losses_dict = vime_training(x_full, x_test, model, device, p_m=config['p_m'], alpha=config['alpha'],
                                         lr=args.ssl_lr)

        if config['freeze_weights'] == 'True' or config['freeze_weights'] is True:
            # /!\ If we used self-supervised learning, we freeze all but the last layer of the encoder
            for name, param in list(model.encoder.named_parameters())[:-2]:
                param.requires_grad = False
    # ===========================================================================

    # ========================== Step 2 - Joint learning ========================
    logging.info('Starting joint learning...')
    losses_dict, model = joint_training(model, x_full, y_train_classifier, x_unlab, y_unlab, x_test, y_test,
                                        y_test_classifier, grouped_unknown_class_val, cat_columns_indexes, None, config, device)
    # ===========================================================================

    # Save the training metrics curves
    fig_path = os.path.join('.', 'figures', 'training_curves', str(args.dataset_name) + '_joint_training_curve.svg')
    logging.info("Saving the metrics training curves in " + fig_path)
    plot_alternative_joint_learning_metrics(losses_dict, fig_path)

    logging.info("Final test metrics:")
    logging.info("BACC = {:.3f} | ACC = {:.3f} | NMI = {:.3f} | ARI = {:.3f}".format(losses_dict['balanced_test_clustering_accuracy'][-1], losses_dict['test_clustering_accuracy'][-1], losses_dict['test_nmi'][-1], losses_dict['test_ari'][-1]))
