from itertools import combinations
from tqdm import tqdm
import time

from src.loss_functions import unsupervised_classification_loss
from src.ncl_memory_module import NCLMemoryModule
from src.transforms import *
from src.utils import *


def joint_training(model, x_full, y_train_classifier, x_unlab, y_unlab, x_test, y_test, y_test_classifier,
                   grouped_unknown_class_val, cat_columns_indexes, cat_features_mask, config, device):
    """
    The joint training phase of the model. This is where the unknown classes are discovered in *x_unlab*.
    :param model: torch.nn.Module: The model to evaluate.
    :param x_full: torch.tensor: The concatenation of x_train and x_unlab.
    :param y_train_classifier: np.array: The concatenation of y_train and y_unlab, where:
            1) all the values of y_unlab are replaced with *grouped_unknown_class_val*.
            and 2) the values of y_train were mapped to [0, |C^l|] using the inverse of *classifier_mapping_dict*.
    :param x_unlab: torch.tensor: The unknown training data.
    :param y_unlab: np.array: The unknown training labels.
    :param x_test: torch.tensor: The testing data.
    :param y_test: np.array: The testing labels (containing both known and unknown).
    :param y_test_classifier: np.array: The values of y_test were mapped to [0, |C^l|] using the inverse of *classifier_mapping_dict*.
    :param grouped_unknown_class_val: int: The value used to replace the classes of y_unlab in y_train_classifier.
    :param cat_columns_indexes: torch.tensor: The indexes of the one-hot categorical columns of the dataset.
    :param cat_features_mask: torch.tensor: Same as *cat_columns_indexes* but as a boolean mask.
    :param config: dict: The dictionary containing the different values of the hyper-parameters.
    :param device: torch.device: The device to send the dataset to.
    :return:
        losses_dict: dict: For every epoch, the metrics are collected and added in this dictionary.
        model: torch.nn.Module: The trained model.
    """
    # Compute the top_k from the percentage of the configuration :
    if config['pseudo_labels_method'] == 'top_k_cosine_faster':
        max_topk = (config['batch_size'] * (config['batch_size'] - 1)) / 2
        computed_topk = int(max_topk * (config['cosine_topk'] / 100))
        logging.info("Computed topk =" + str(computed_topk) + "(" + str(config['cosine_topk']) + "% of " + str(max_topk) + ")")
    elif config['pseudo_labels_method'] == 'top_k_cosine_per_instance':
        max_topk = config['batch_size'] / 2
        computed_topk = int(max_topk * (config['cosine_topk'] / 100))
        logging.info("Computed topk =" + str(computed_topk) + "(" + str(config['cosine_topk']) + "% of " + str(max_topk) + ")")

    losses_dict = {
        # Losses
        'train_classification_losses': [],
        'train_clustering_losses': [],
        'bce_losses': [],
        'ce_losses': [],
        'train_cs_classification_losses': [],
        'train_cs_clustering_losses': [],

        # Performance metrics
        'train_ari': [],
        'test_ari': [],
        'train_nmi': [],
        'test_nmi': [],
        'train_classification_accuracy': [],
        'test_classification_accuracy': [],
        'train_clustering_accuracy': [],
        'test_clustering_accuracy': [],
        'balanced_train_clustering_accuracy': [],
        'balanced_test_clustering_accuracy': []
    }

    unlab_memory_module = NCLMemoryModule(device, M=config['M'], labeled_memory=False)
    lab_memory_module = NCLMemoryModule(device, M=config['M'], labeled_memory=True)

    optimizer_classification = torch.optim.AdamW(model.parameters(), lr=config['lr_classif'])
    optimizer_clustering = torch.optim.AdamW(model.parameters(), lr=config['lr_cluster'])

    for epoch in range(config['epochs']):
        n_batchs = math.ceil((x_full.shape[0]) / config['batch_size'])

        cross_entropy_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        train_classification_losses = []
        train_clustering_losses = []
        train_bce_losses = []
        train_ce_losses = []
        train_cs_classification_losses = []
        train_cs_clustering_losses = []

        with tqdm(range(n_batchs)) as t:
            t.set_description(str(epoch + 1) + "/" + str(config['epochs']))

            batch_start_index, batch_end_index = 0, config['batch_size']
            for batch_index in range(n_batchs):
                # (1) ===== Get the data =====
                batch_x_train = x_full[batch_start_index:batch_end_index]
                batch_y_train = y_train_classifier[batch_start_index:batch_end_index]

                mask_unlab = batch_y_train == grouped_unknown_class_val
                mask_lab = ~mask_unlab
                assert mask_unlab.sum() > 0, "No unlabeled data in batch"

                # augment/transform the data
                with torch.no_grad():  # <= /!\ /!\ /!\
                    if config['transform_method'] == 'old':
                        augmented_x_unlab = transform_batch(batch_x_train[mask_unlab], unlab_memory_module.original_data_memory, device)
                        augmented_x_lab = transform_batch(batch_x_train[mask_lab], lab_memory_module.original_data_memory, device)
                    elif config['transform_method'] == 'old_2':
                        augmented_x_unlab = transform_batch_vectorized(batch_x_train[mask_unlab], unlab_memory_module.original_data_memory, device)
                        augmented_x_lab = transform_batch_vectorized(batch_x_train[mask_lab],
                                                                     lab_memory_module.original_data_memory, device)
                    elif config['transform_method'] == 'new_1':
                        augmented_x_unlab = smotenc_transform_batch(batch_x_train[mask_unlab], cat_features_mask, unlab_memory_module.original_data_memory, device)
                        augmented_x_lab = smotenc_transform_batch(batch_x_train[mask_lab], cat_features_mask, lab_memory_module.original_data_memory, device)
                    elif config['transform_method'] == 'new_2':
                        augmented_x_unlab = smotenc_transform_batch_2(batch_x_train[mask_unlab], cat_columns_indexes, unlab_memory_module.original_data_memory, device, k_neighbors=config['k_neighbors'])
                        augmented_x_lab = smotenc_transform_batch_2(batch_x_train[mask_lab], cat_columns_indexes, lab_memory_module.original_data_memory, device, k_neighbors=config['k_neighbors'])
                    else:
                        raise ValueError('Undefined data augmentation method : ' + str(config['transform_method']))

                # (2) ===== Forward the classification data and compute the losses =====
                encoded_x = model.encoder_forward(batch_x_train)
                encoded_augmented_x_unlab = model.encoder_forward(augmented_x_unlab)
                encoded_augmented_x_lab = model.encoder_forward(augmented_x_lab)
                y_pred_lab = model.classification_head_forward(encoded_x)
                augmented_y_pred = torch.zeros(y_pred_lab.shape, device=device)
                augmented_y_pred[mask_lab] = model.classification_head_forward(encoded_augmented_x_lab)
                augmented_y_pred[mask_unlab] = model.classification_head_forward(encoded_augmented_x_unlab)

                ce_loss = cross_entropy_loss(y_pred_lab, torch.tensor(batch_y_train, device=device))

                cs_loss_classifier = mse_loss(y_pred_lab, augmented_y_pred)

                classifier_loss = config['w1'] * ce_loss + (1 - config['w1']) * cs_loss_classifier

                # backward
                optimizer_classification.zero_grad()
                classifier_loss.backward()
                optimizer_classification.step()

                # (3) ===== Forward the clustering data and compute the losses =====
                encoded_x_unlab = model.encoder_forward(batch_x_train[mask_unlab])
                y_pred_unlab = model.clustering_head_forward(encoded_x_unlab)

                encoded_augmented_x_unlab = model.encoder_forward(augmented_x_unlab)
                augmented_y_pred_unlab = model.clustering_head_forward(encoded_augmented_x_unlab)

                # Define the pseudo labels
                if config['pseudo_labels_method'] == 'cosine':
                    upper_list_1, upper_list_2 = np.triu_indices(len(encoded_x_unlab), k=1)
                    unlab_unlab_similarities = nn.CosineSimilarity()(encoded_x_unlab[upper_list_1],
                                                                     encoded_x_unlab[upper_list_2])
                    pseudo_labels = (unlab_unlab_similarities > config['cosine_topk']).to(torch.int32)
                    bce_loss = unsupervised_classification_loss(y_pred_unlab[upper_list_1],
                                                                y_pred_unlab[upper_list_2],
                                                                pseudo_labels)
                elif config['pseudo_labels_method'] == 'top_k_cosine':
                    unlab_unlab_combinations_to_compute = np.array(
                        [list(comb) for comb in combinations(list(range(len(encoded_x_unlab))), 2)])
                    unlab_unlab_similarities = nn.CosineSimilarity()(
                        encoded_x_unlab[unlab_unlab_combinations_to_compute[:, 0]],
                        encoded_x_unlab[unlab_unlab_combinations_to_compute[:, 1]])
                    pseudo_labels = torch.zeros(len(unlab_unlab_similarities), device=device)
                    pseudo_labels[unlab_unlab_similarities.argsort(descending=True)[:computed_topk]] = 1
                    bce_loss = unsupervised_classification_loss(y_pred_unlab[unlab_unlab_combinations_to_compute[:, 0]],
                                                                y_pred_unlab[unlab_unlab_combinations_to_compute[:, 1]],
                                                                pseudo_labels)
                elif config['pseudo_labels_method'] == 'top_k_cosine_faster':
                    upper_list_1, upper_list_2 = np.triu_indices(len(encoded_x_unlab), k=1)
                    unlab_unlab_similarities = nn.CosineSimilarity()(encoded_x_unlab[upper_list_1],
                                                                     encoded_x_unlab[upper_list_2])
                    pseudo_labels = torch.zeros(len(unlab_unlab_similarities), device=device)
                    pseudo_labels[unlab_unlab_similarities.argsort(descending=True)[:computed_topk]] = 1
                    bce_loss = unsupervised_classification_loss(y_pred_unlab[upper_list_1],
                                                                y_pred_unlab[upper_list_2],
                                                                pseudo_labels)
                elif config['pseudo_labels_method'] == 'top_k_cosine_per_instance':
                    # Because it is symmetric, we compute the upper corner and copy it to the lower corner
                    upper_list_1, upper_list_2 = np.triu_indices(len(encoded_x_unlab), k=1)
                    unlab_unlab_similarities = nn.CosineSimilarity()(encoded_x_unlab[upper_list_1],
                                                                     encoded_x_unlab[upper_list_2])
                    similarity_matrix = torch.zeros((len(encoded_x_unlab), len(encoded_x_unlab)), device=device)
                    similarity_matrix[upper_list_1, upper_list_2] = unlab_unlab_similarities
                    similarity_matrix += similarity_matrix.T.clone()

                    # Get for each instance the cosine_topk most similar
                    top_k_most_similar_instances_per_instance = similarity_matrix.argsort(descending=True)[:,
                                                                :computed_topk]

                    pseudo_labels_matrix = torch.zeros((len(encoded_x_unlab), len(encoded_x_unlab)), device=device)
                    pseudo_labels_matrix = pseudo_labels_matrix.scatter_(
                        index=top_k_most_similar_instances_per_instance, dim=1, value=1)

                    # The matrix isn't symmetric, because the graph is directed
                    # So if there is one link between two points, regardless of the direction, we consider this pair to be positive
                    pseudo_labels_matrix += pseudo_labels_matrix.T.clone()
                    pseudo_labels_matrix[pseudo_labels_matrix > 1] = 1  # Some links will overlap

                    pseudo_labels = pseudo_labels_matrix[upper_list_1, upper_list_2]
                    bce_loss = unsupervised_classification_loss(y_pred_unlab[upper_list_1],
                                                                y_pred_unlab[upper_list_2],
                                                                pseudo_labels)
                elif config['pseudo_labels_method'] == 'ranking':
                    target_ulb = ranking_stats_pseudo_labels(encoded_x_unlab.detach(), device, topk=5)
                    prob1_ulb, _ = PairEnum(y_pred_unlab)
                    _, prob2_ulb = PairEnum(augmented_y_pred_unlab)
                    bce_loss = unsupervised_classification_loss(prob1_ulb, prob2_ulb, target_ulb)
                else:
                    raise ValueError('Undefined pseudo labels method : ' + str(config['pseudo_labels_method']))

                cs_loss_clustering = mse_loss(y_pred_unlab, augmented_y_pred_unlab)

                clustering_loss = config['w2'] * bce_loss + (1 - config['w2']) * cs_loss_clustering

                # backward
                optimizer_clustering.zero_grad()
                clustering_loss.backward()
                optimizer_clustering.step()

                # Save losses for plotting purposes
                train_classification_losses.append(classifier_loss.item())
                train_clustering_losses.append(clustering_loss.item())
                train_bce_losses.append(bce_loss.item())
                train_ce_losses.append(ce_loss.item())
                train_cs_classification_losses.append(cs_loss_classifier.item())
                train_cs_clustering_losses.append(cs_loss_clustering.item())

                t.set_postfix_str("classif={:05.3f}".format(pretty_mean(train_classification_losses))
                                  + " clust={:05.3f}".format(pretty_mean(train_clustering_losses))
                                  + " ce={:05.3f}".format(pretty_mean(train_ce_losses))
                                  + " bce={:05.3f}".format(pretty_mean(train_bce_losses))
                                  + " cs1={:05.3f}".format(pretty_mean(train_cs_classification_losses))
                                  + " cs2={:05.3f}".format(pretty_mean(train_cs_clustering_losses)))

                # update the memory modules
                unlab_memory_module.memory_step(encoded_x_unlab.detach().clone(),
                                                batch_x_train[mask_unlab].detach().clone())
                lab_memory_module.memory_step(encoded_x[mask_lab].detach().clone(),
                                              batch_x_train[mask_lab].detach().clone(),
                                              input_labels=torch.tensor(batch_y_train[mask_lab], device=device))

                t.update()
                batch_start_index += config['batch_size']
                batch_end_index = min((batch_end_index + config['batch_size']), x_full.shape[0])

        train_classification_accuracy = compute_classification_accuracy(x_full, y_train_classifier, y_train_classifier,
                                                                        model)
        test_classification_accuracy = compute_classification_accuracy(x_test, y_test_classifier, y_train_classifier,
                                                                       model)

        train_clustering_accuracy = compute_clustering_accuracy(x_unlab, y_unlab, y_unlab, model)
        test_clustering_accuracy = compute_clustering_accuracy(x_test, y_test, y_unlab, model)

        balanced_train_clustering_accuracy = compute_balanced_clustering_accuracy(x_unlab, y_unlab, y_unlab, model)
        balanced_test_clustering_accuracy = compute_balanced_clustering_accuracy(x_test, y_test, y_unlab, model)

        train_ari, train_nmi = compute_ari_and_nmi(x_unlab, y_unlab, y_unlab, model)
        test_ari, test_nmi = compute_ari_and_nmi(x_test, y_test, y_unlab, model)

        losses_dict['train_ari'].append(train_ari)
        losses_dict['test_ari'].append(test_ari)
        losses_dict['train_nmi'].append(train_nmi)
        losses_dict['test_nmi'].append(test_nmi)
        losses_dict['train_classification_losses'].append(np.mean(train_classification_losses))
        losses_dict['train_clustering_losses'].append(np.mean(train_clustering_losses))
        losses_dict['bce_losses'].append(np.mean(train_bce_losses))
        losses_dict['ce_losses'].append(np.mean(train_ce_losses))
        losses_dict['train_cs_classification_losses'].append(np.mean(train_cs_classification_losses))
        losses_dict['train_cs_clustering_losses'].append(np.mean(train_cs_clustering_losses))
        losses_dict['train_classification_accuracy'].append(train_classification_accuracy)
        losses_dict['test_classification_accuracy'].append(test_classification_accuracy)
        losses_dict['test_clustering_accuracy'].append(test_clustering_accuracy)
        losses_dict['train_clustering_accuracy'].append(train_clustering_accuracy)
        losses_dict['balanced_test_clustering_accuracy'].append(balanced_test_clustering_accuracy)
        losses_dict['balanced_train_clustering_accuracy'].append(balanced_train_clustering_accuracy)

        logging.debug("Train / Test clustering accuracy = {:05.3f} / {:05.3f}".format(train_clustering_accuracy,
                                                                                      test_clustering_accuracy))
        logging.debug(
            "Train / Test balanced clustering accuracy = {:05.3f} / {:05.3f}".format(balanced_train_clustering_accuracy,
                                                                                     balanced_test_clustering_accuracy))
        time.sleep(0.5)  # for tqdm pretty prints
    # ===========================================================================

    torch.cuda.empty_cache()  # Free some memory up

    return losses_dict, model


def vime_training(x_vime, x_test, model, device, p_m=0.3, alpha=2.0, lr=0.001, num_epochs=30, batch_size=128):
    """
    The encoder's initialization phase.
    :param x_vime: torch.tensor: torch.tensor: The full training data, known or unknown.
    :param x_test: torch.tensor: torch.tensor: The full testing data, known or unknown.
    :param model: torch.nn.Module, the torch model to train.
    :param device: torch.device : The device.
    :param p_m: float: Corruption probability
    :param alpha: Loss_vime = mask_estim_loss + alpha * feature_estim_loss
    :param lr: float: Learning rate of the self-supervised learning phase.
    :param num_epochs: int: The number of epochs to train.
    :param batch_size: int: The batch size.
    :return:
        losses_dict: dict: For every epoch, the metrics are collected and added in this dictionary.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses_dict = {
        'epoch_mean_train_losses': [],
        'epoch_mean_test_losses': [],
        'epoch_mean_lr_projection_score': [],
        'lr_base_projection_score': None
    }

    for epoch in range(num_epochs):
        n_batchs = math.ceil((x_vime.shape[0]) / batch_size)
        with tqdm(range(n_batchs)) as t:
            t.set_description("Epoch " + str(epoch + 1) + " / " + str(num_epochs))
            train_losses = []
            mask_losses = []
            feature_losses = []

            batch_start_index, batch_end_index = 0, min(batch_size, len(x_vime))
            for batch_index in range(n_batchs):
                batch_X_train = x_vime[batch_start_index:batch_end_index]

                m_unlab = np.random.binomial(1, p_m, batch_X_train.shape)
                m_label, x_tilde = pretext_generator(m_unlab, batch_X_train.to('cpu').numpy())

                x_tilde = torch.Tensor(x_tilde).to(device)
                m_label = torch.Tensor(m_label).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                mask_pred, feature_pred = model.vime_forward(x_tilde)

                # compute losses
                mask_loss, feature_loss = vime_loss(mask_pred, m_label, feature_pred, batch_X_train)
                loss = mask_loss + alpha * feature_loss

                # backward
                loss.backward()

                # update the weights using gradient descent
                optimizer.step()

                # Save loss for plotting purposes
                train_losses.append(loss.item())
                mask_losses.append(mask_loss.item())
                feature_losses.append(feature_loss.item())

                # print statistics
                t.set_postfix_str("loss={:05.3f}".format(np.mean(train_losses)) +
                                  " - mask_loss={:05.3f}".format(np.mean(mask_losses)) +
                                  " - feature_loss={:05.3f}".format(np.mean(feature_losses)))
                t.update()
                batch_start_index += batch_size
                batch_end_index = min((batch_end_index + batch_size), x_vime.shape[0])

        # Evaluate on the test set
        test_mask_loss, test_feature_loss = evaluate_vime_model_on_set(x_test, model, device)
        test_loss = test_mask_loss + alpha * test_feature_loss

        losses_dict['epoch_mean_train_losses'].append(np.mean(train_losses))
        losses_dict['epoch_mean_test_losses'].append(test_loss.item())

    return losses_dict
