import torch.nn.functional as F
import torch.nn as nn
import random
import torch
import math


def transform_batch(batch, data_queue, device):
    """
    Slow but easily understandable non-vectorized transformation for *numerical* data only.
    Inspired from SMOTE.
    :param batch: torch.tensor : The batch data to transform.
    :param data_queue: torch.tensor: The labelled data stored in the lab_memory_module object.
    :param device: torch.device : The device.
    :return: torch.Tensor: The transformed data.
    """
    full_data = torch.cat([batch, data_queue])
    transformed_batch = torch.tensor([], device=device, dtype=torch.float32)

    for i in range(len(batch)):
        # Ignore duplicates and current point
        full_data_mask = [j for j in range(len(full_data))]  # if not torch.equal(full_data[j], full_data[i])
        full_data_mask.remove(i)

        # similarities = euclidean_distances(X=batch[i].reshape(1, -1), Y=full_data[full_data_mask])[0]
        similarities = nn.CosineSimilarity()(batch[i].repeat(len(full_data_mask), 1), full_data[full_data_mask])

        closest_point_index = full_data_mask[similarities.argmax()]

        diff_vect = (full_data[closest_point_index] - batch[i]) * random.uniform(0, 1)

        transformed_batch = torch.cat([transformed_batch, (batch[i] + diff_vect).view(1, -1)])

    return transformed_batch


def transform_batch_vectorized(batch, data_queue, device, batch_size=100):
    """
    Faster but harder to understand vectorized transformation for *numerical* data only.
    Inspired from SMOTE.
    See 'transform_batch' to really understand the logic.
    :param batch: torch.tensor: The batch data to transform.
    :param data_queue: torch.tensor: The labelled data stored in the lab_memory_module object.
    :param device: torch.device: The device.
    :param batch_size: int: During computation, the batch is cut in blocs of size batch_size. If you have memory errors, reduce it.
    :return: torch.Tensor: The transformed data.
    """
    full_data = torch.cat([batch, data_queue])

    full_similarities_matrix = torch.tensor([], device=device, dtype=torch.long)
    n_batchs = math.ceil((full_data.shape[0]) / batch_size)
    batch_start_index, batch_end_index = 0, min(batch_size, len(full_data))
    for batch_index in range(n_batchs):
        batch_similarities = F.cosine_similarity(batch.unsqueeze(1), full_data[batch_start_index:batch_end_index], dim=-1)
        full_similarities_matrix = torch.cat([full_similarities_matrix, batch_similarities], dim=1)

        batch_start_index += batch_size
        batch_end_index = min((batch_end_index + batch_size), full_data.shape[0])

    full_similarities_matrix -= torch.eye(len(batch), len(full_data), device=device)  # This way, itself wont be in the most similar instances

    most_similar_indexes = full_similarities_matrix.argmax(dim=1)

    batch_diff_vect = (full_data[most_similar_indexes] - batch) * torch.rand(len(batch), device=device).view(-1, 1)

    return batch + batch_diff_vect


def smotenc_transform_batch(batch, cat_features_mask, data_queue, device, k_neighbors=5, dist='cosine'):
    """
    Slow but easily understandable non-vectorized transformation for mixed numerical and categorical data.
    Inspired from SMOTE-NC.
    :param batch: torch.tensor: The batch data to transform.
    :param cat_features_mask: Array-like object of length (x.shape[1]). Where each element is False if this column is numerical and True if categorical.
    :param data_queue: torch.tensor: The unlabelled data stored in the unlab_memory_module object.
    :param device: torch.device: The device.
    :param k_neighbors: int: The number of neighbors to consider during the transformation.
    :param dist: The distance metric to use. Choices : ['cosine', 'euclidean'].
    :return: torch.tensor: The transformed data.
    """
    full_data = torch.cat([batch, data_queue])
    transformed_batch = torch.tensor([], device=device, dtype=torch.float32)

    for i in range(len(batch)):
        full_data_mask = torch.arange(len(full_data))
        full_data_mask = full_data_mask[full_data_mask != i]

        if dist == 'cosine':
            similarities = nn.CosineSimilarity()(batch[i].repeat(len(full_data_mask), 1), full_data[full_data_mask])
            topk_res = similarities.topk(k=k_neighbors)
            topk_similar_indexes = topk_res.indices
        elif dist == 'euclidean':
            similarities = torch.cdist(batch[i].view(1, -1), full_data[full_data_mask])
            topk_res = similarities.topk(k=k_neighbors, largest=False)
            topk_similar_indexes = topk_res.indices

        closest_points_indexes = full_data_mask[topk_similar_indexes]  # indexes in masked data

        # Select a random point between the k closest
        closest_point_index = closest_points_indexes[random.randrange(k_neighbors)]

        diff_vect = (full_data[closest_point_index] - batch[i]) * torch.rand(1, device=device)
        augmented_vect = batch[i] + diff_vect  # At this point, the categorical values are wrong

        cat_columns_indexes = torch.arange(len(cat_features_mask))[cat_features_mask]
        augmented_vect[cat_columns_indexes] = full_data[closest_points_indexes][:, cat_columns_indexes].mode(0).values

        transformed_batch = torch.cat([transformed_batch, augmented_vect.view(1, -1)])

    return transformed_batch


def smotenc_transform_batch_2(batch, cat_columns_indexes, data_queue, device, k_neighbors=5, dist='cosine', batch_size=100):
    """
    Faster but harder to understand vectorized transformation for mixed numerical and categorical data.
    See 'smotenc_transform_batch' to really understand the logic.
    Inspired from SMOTE-NC.
    :param batch: torch.tensor: The batch data to transform.
    :param cat_columns_indexes: Array-like object of the indexes of the categorical columns. Only useful when transform_method='new_2'.
    :param data_queue: The unlabelled data stored in the unlab_memory_module object.
    :param device: torch.device: The device.
    :param k_neighbors: int: The number of neighbors to consider during the transformation.
    :param dist: The distance metric to use. Choices: ['cosine', 'euclidean'].
    :param batch_size: int: During computation, the batch is cut in blocs of size batch_size. If you have memory errors, reduce it.
    :return: torch.tensor: The transformed data.
    """
    full_data = torch.cat([batch, data_queue])

    full_similarities_matrix = torch.tensor([], device=device, dtype=torch.long)

    n_batchs = math.ceil((full_data.shape[0]) / batch_size)
    batch_start_index, batch_end_index = 0, min(batch_size, len(full_data))
    for batch_index in range(n_batchs):
        if dist == 'cosine':
            similarities = F.cosine_similarity(batch.unsqueeze(1), full_data[batch_start_index:batch_end_index], dim=-1)
        elif dist == 'euclidean':
            # ToDo (below is the non-vectorized code)
            # similarities = torch.cdist(batch[i].view(1, -1), full_data)
            # similarities[i] += torch.inf  # This way, itself wont be in the most similar instances
            # topk_similar_indexes = similarities.topk(k=k_neighbors, largest=False).indices
            pass

        full_similarities_matrix = torch.cat([full_similarities_matrix, similarities], dim=1)

        batch_start_index += batch_size
        batch_end_index = min((batch_end_index + batch_size), full_data.shape[0])

    full_similarities_matrix -= torch.eye(len(batch), len(full_data), device=device)  # This way, itself wont be in the most similar instances

    batch_topk_similar_indexes = full_similarities_matrix.topk(k=k_neighbors, dim=1).indices

    # Select a random point between the k closest
    batch_closest_point_index = torch.gather(batch_topk_similar_indexes, 1, torch.randint(low=0, high=k_neighbors, size=(len(batch),), device=device).view(-1, 1))
    batch_closest_point_index = batch_closest_point_index.flatten()

    batch_closest_point = full_data[batch_closest_point_index]

    batch_diff_vect = (batch_closest_point - batch) * torch.rand(len(batch), device=device).view(-1, 1)

    augmented_batch = batch + batch_diff_vect  # At this point, the categorical values are wrong, next line fixes that

    augmented_batch[:, cat_columns_indexes] = full_data[:, cat_columns_indexes.flatten()][batch_topk_similar_indexes].mode(1).values

    return augmented_batch
