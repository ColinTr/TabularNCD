from keras.datasets import mnist
import pandas as pd
import numpy as np
import logging
import random
import torch


def import_dataset_with_name(dataset_name, device):
    """
    Import procedure of the individual datasets.
    This is where the number of unknown classes is defined for each dataset.
    :param dataset_name: ToDo
    :param device: ToDo
    :return: ToDo
    """
    if dataset_name == 'mnist':
        n_unknown_classes = 5

        # Import the dataset
        x_train, y_train, x_unlab, y_unlab, x_test, y_test = import_split_mnist(device, n_unknown_classes=n_unknown_classes, seed=0, shuffle=False)

        # Split and create the subsets needed for training
        x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict = complete_import_procedure(x_train, y_train, x_unlab, y_unlab, x_test, y_test)

        cat_columns_indexes = torch.tensor([], device=device, dtype=torch.int64)
    elif dataset_name == 'ForestCoverType':
        n_unknown_classes = 3

        # Import the dataset
        x_train, y_train, x_unlab, y_unlab, x_test, y_test = import_split_dataset('./data/' + dataset_name + '/', device, n_unknown_classes=n_unknown_classes)

        # Split and create the subsets needed for training
        x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict = complete_import_procedure(x_train, y_train, x_unlab, y_unlab, x_test, y_test)

        cat_columns_indexes = torch.tensor(list(range(10, 54)), device=device, dtype=torch.int64)

        # ForestCoverType dataset has too many test instances
        x_test = x_test[:50000]
        y_test = y_test[:50000]
        y_test_classifier = y_test_classifier[:50000]
    elif dataset_name == 'Abalone':
        n_unknown_classes = 7

        # Import the dataset
        x_train, y_train, x_unlab, y_unlab, x_test, y_test = import_split_dataset('./data/' + dataset_name + '/', device, n_unknown_classes=n_unknown_classes)

        # Split and create the subsets needed for training
        x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict = complete_import_procedure(x_train, y_train, x_unlab, y_unlab, x_test, y_test)

        cat_columns_indexes = torch.tensor([7, 8, 9], device=device, dtype=torch.int64)
    elif dataset_name == 'LetterRecognition':
        n_unknown_classes = 7

        # Import the dataset
        x_train, y_train, x_unlab, y_unlab, x_test, y_test = import_split_dataset('./data/' + dataset_name + '/', device, n_unknown_classes=n_unknown_classes)

        # Split and create the subsets needed for training
        x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict = complete_import_procedure(x_train, y_train, x_unlab, y_unlab, x_test, y_test)

        cat_columns_indexes = torch.tensor([], device=device, dtype=torch.int64)
    elif dataset_name == 'HumanActivityRecognition':
        n_unknown_classes = 3

        # Import the dataset
        x_train, y_train, x_unlab, y_unlab, x_test, y_test = import_split_dataset('./data/' + dataset_name + '/', device, n_unknown_classes=n_unknown_classes)

        # Split and create the subsets needed for training
        x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict = complete_import_procedure(x_train, y_train, x_unlab, y_unlab, x_test, y_test)

        cat_columns_indexes = torch.tensor([], device=device, dtype=torch.int64)
    elif dataset_name == 'Satimage':
        n_unknown_classes = 3

        # Import the dataset
        x_train, y_train, x_unlab, y_unlab, x_test, y_test = import_split_dataset('./data/' + dataset_name + '/', device, n_unknown_classes=n_unknown_classes)

        # Split and create the subsets needed for training
        x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict = complete_import_procedure(x_train, y_train, x_unlab, y_unlab, x_test, y_test)

        cat_columns_indexes = torch.tensor([], device=device, dtype=torch.int64)
    elif dataset_name == 'Pendigits':
        n_unknown_classes = 5

        # Import the dataset
        x_train, y_train, x_unlab, y_unlab, x_test, y_test = import_split_dataset('./data/' + dataset_name + '/', device, n_unknown_classes=n_unknown_classes)

        # Split and create the subsets needed for training
        x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict = complete_import_procedure(x_train, y_train, x_unlab, y_unlab, x_test, y_test)

        cat_columns_indexes = torch.tensor([], device=device, dtype=torch.int64)
    elif dataset_name == 'USCensus1990':
        n_unknown_classes = 6

        # Import the dataset
        x_train, y_train, x_unlab, y_unlab, x_test, y_test = import_split_dataset('./data/' + dataset_name + '/', device, n_unknown_classes=n_unknown_classes)

        # Otherwise there are too many data points to handle
        x_train, y_train = x_train[:50000], y_train[:50000]
        x_unlab, y_unlab = x_unlab[:50000], y_unlab[:50000]
        x_test, y_test = x_test[:50000], y_test[:50000]

        # Split and create the subsets needed for training
        x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict = complete_import_procedure(x_train, y_train, x_unlab, y_unlab, x_test, y_test)

        cat_columns_indexes = torch.tensor([], device=device, dtype=torch.int64)
    elif dataset_name == 'ImageSegmentation':
        n_unknown_classes = 5

        # Import the dataset
        x_train, y_train, x_unlab, y_unlab, x_test, y_test = import_split_dataset('./data/' + dataset_name + '/', device, n_unknown_classes=n_unknown_classes)

        # Split and create the subsets needed for training
        x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict = complete_import_procedure(x_train, y_train, x_unlab, y_unlab, x_test, y_test)

        cat_columns_indexes = torch.tensor([], device=device, dtype=torch.int64)
    elif dataset_name == 'CNAE-9':
        n_unknown_classes = 5

        # Import the dataset
        x_train, y_train, x_unlab, y_unlab, x_test, y_test = import_split_dataset('./data/' + dataset_name + '/', device, n_unknown_classes=n_unknown_classes)

        # Split and create the subsets needed for training
        x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict = complete_import_procedure(x_train, y_train, x_unlab, y_unlab, x_test, y_test)

        cat_columns_indexes = torch.tensor([], device=device, dtype=torch.int64)
    else:
        raise ValueError('Unknown dataset : ' + dataset_name)

    return x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict, cat_columns_indexes


def import_split_mnist(device, n_unknown_classes, seed=0, shuffle=True):
    """
    Import the mnist data set and split it into the three subsets : train, unlab and test.
    The values are normalized to range from 0 to 1 by dividing the original values by 255.0.
    The images are flattened to create instances of 28*28=784 features.
    The classes of train and unlab are disjoint. The test set contains all of the classes.

    :param device: The device to send the dataset to, torch.device.
    :param n_unknown_classes: int : The number of unknown classes to define (between 1 and 9), int.
    :param seed: int : Random seed for reproducibility of the random splits, int.
    :param shuffle: bool : Shuffle the classes or not.
    :return: x_train, y_train, x_unlab, y_unlab, x_test, y_test : torch.tensor(s)
    """
    x_train_, y_train_, x_test_, y_test_ = import_mnist()

    # Select and separate the known and unknown classes
    classes = np.unique(y_train_)

    if shuffle is True:
        random.seed(seed)  # For reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        np.random.shuffle(classes)

    known_classes = classes[:len(classes) - n_unknown_classes]
    unknown_classes = classes[len(classes) - n_unknown_classes:]

    x_train = torch.tensor(x_train_[np.in1d(y_train_, known_classes)], dtype=torch.float, device=device)
    y_train = y_train_[np.in1d(y_train_, known_classes)]

    x_unlab = torch.tensor(x_train_[np.in1d(y_train_, unknown_classes)], dtype=torch.float, device=device)
    y_unlab = y_train_[np.in1d(y_train_, unknown_classes)]

    x_test = torch.tensor(x_test_, dtype=torch.float, device=device)

    logging.debug("x_train shape = " + str(list(x_train.shape)) + " & y_train shape = " + str(list(y_train.shape)) + " with " + str(len(np.unique(y_train))) + " unique classes")
    logging.debug("x_unlab shape = " + str(list(x_unlab.shape)) + " & y_unlab shape = " + str(list(y_unlab.shape)) + " with " + str(len(np.unique(y_unlab))) + " unique classes")
    logging.debug("x_test shape = " + str(list(x_test.shape)) + " & y_test shape = " + str(list(y_test_.shape)) + " with " + str(len(np.unique(y_test_))) + " unique classes")

    return x_train, y_train, x_unlab, y_unlab, x_test, y_test_


def import_mnist():
    """
    Simple import of the MNIST data set in tabular form.
    The values are normalized to range from 0 to 1 by dividing the original values by 255.0.
    The images are flattened to create instances of 28*28=784 features.

    :return: x_train, y_train, x_test, y_test : np.array(s).
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize features (scale between 0 and 1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Treat MNIST data as tabular data with 784 features
    no, dim_x, dim_y = np.shape(x_train)
    test_no, _, _ = np.shape(x_test)

    x_train = np.reshape(x_train, [no, dim_x * dim_y])
    x_test = np.reshape(x_test, [test_no, dim_x * dim_y])

    return x_train, y_train, x_test, y_test


def import_split_dataset(dataset_folder_path, device, n_unknown_classes, seed=0, shuffle=True):
    """
    ToDo
    :param dataset_folder_path: ToDo
    :param device: ToDo
    :param n_unknown_classes: ToDo
    :param seed: ToDo
    :param shuffle: ToDo
    :return: ToDo
    """
    train_df = pd.read_csv(dataset_folder_path + 'treated_train.csv')
    test_df = pd.read_csv(dataset_folder_path + 'treated_test.csv')

    x_train_ = np.array(train_df.drop(['classes'], axis=1))
    y_train_ = np.array(train_df['classes'])

    x_test_ = np.array(test_df.drop(['classes'], axis=1))
    y_test_ = np.array(test_df['classes'])

    # Select and separate the known and unknown classes
    classes = np.unique(y_train_)

    if shuffle is True:
        random.seed(seed)  # For reproducibility
        np.random.seed(seed)
        np.random.shuffle(classes)

    known_classes = classes[:len(classes) - n_unknown_classes]
    unknown_classes = classes[len(classes) - n_unknown_classes:]

    x_train = torch.tensor(x_train_[np.in1d(y_train_, known_classes)], dtype=torch.float, device=device)
    y_train = y_train_[np.in1d(y_train_, known_classes)]

    x_unlab = torch.tensor(x_train_[np.in1d(y_train_, unknown_classes)], dtype=torch.float, device=device)
    y_unlab = y_train_[np.in1d(y_train_, unknown_classes)]

    x_test = torch.tensor(x_test_, dtype=torch.float, device=device)

    logging.info("x_train shape = " + str(list(x_train.shape)) + " & y_train shape = " + str(list(y_train.shape)) + " with " + str(len(np.unique(y_train))) + " unique classes")
    logging.info("x_unlab shape = " + str(list(x_unlab.shape)) + " & y_unlab shape = " + str(list(y_unlab.shape)) + " with " + str(len(np.unique(y_unlab))) + " unique classes")
    logging.info("x_test shape = " + str(list(x_test.shape)) + " & y_test shape = " + str(list(y_test_.shape)) + " with " + str(len(np.unique(y_test_))) + " unique classes")

    return x_train, y_train, x_unlab, y_unlab, x_test, y_test_


def complete_import_procedure(x_train, y_train, x_unlab, y_unlab, x_test, y_test):
    """
    ToDo
    :param x_train: ToDo
    :param y_train: ToDo
    :param x_unlab: ToDo
    :param y_unlab: ToDo
    :param x_test: ToDo
    :param y_test: ToDo
    :return: ToDo
    """
    # We train on both labeled and unlabed data
    x_full = torch.cat([x_train, x_unlab])
    y_full = np.concatenate((y_train, np.repeat(-1, len(x_unlab))), axis=0)

    # So that the classification target labels are in {0; ...; |Cl|+1}
    y_full_classifier = y_full.astype(np.int64).copy()
    y_full_classifier[y_full_classifier == -1] = 999
    classifier_mapper, classifier_ind = np.unique(y_full_classifier, return_inverse=True)
    classifier_mapping_dict = dict(zip(y_full_classifier, classifier_ind))
    y_train_classifier = np.array(list(map(classifier_mapping_dict.get, y_full_classifier)))

    y_test_classifier = y_test.astype(np.int64).copy()
    y_test_classifier[np.in1d(y_test, np.unique(y_unlab))] = 999
    y_test_classifier = np.array(list(map(classifier_mapping_dict.get, y_test_classifier)))

    grouped_unknown_class_val = classifier_mapping_dict[999]

    # Jointly shuffle them
    p = np.random.permutation(len(x_full))
    x_full = x_full[p]
    y_full = y_full[p]
    y_full_classifier = y_full_classifier[p]
    y_train_classifier = y_train_classifier[p]

    logging.info("Known classes :" + str(np.unique(y_train)))
    logging.info("Unknown classes :" + str(np.unique(y_unlab)))
    logging.info("x_train shape :" + str(x_train.shape))
    logging.info("x_unlab shape :" + str(x_unlab.shape))
    logging.info("x_test shape :" + str(x_test.shape))

    logging.info("Share of unlabelled data: {:.2f}".format((len(x_unlab) / (len(x_train) + len(x_unlab))) * 100) + "%")

    return x_train, y_train, x_unlab, y_unlab, x_test, y_test, x_full, y_full, y_full_classifier, y_train_classifier, y_test_classifier, grouped_unknown_class_val, classifier_mapping_dict
