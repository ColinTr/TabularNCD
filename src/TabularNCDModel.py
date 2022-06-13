import torch.nn as nn


class TabularNCDModel(nn.Module):
    def __init__(self, encoder_layers_sizes, ssl_layers_sizes, joint_learning_layers_sizes,
                 n_known_classes, n_unknown_classes, activation_fct, encoder_last_activation_fct,
                 ssl_last_activation_fct, joint_last_activation_fct, p_dropout):
        """
        The TabularNCD model object. It is composed of 5 main networks : The *encoder*, for SSL the *mask_vector_estimator*
        and *feature_vector_estimator* and for the joint learning the *classification_head* and *clustering_head*.
        :param encoder_layers_sizes: Size from the input to the output, not only the hidden layers.
        :param ssl_layers_sizes: Only the hidden layers, as the input and output depend on the encoder and input vector size. This corresponds to the mask and feature vector estimators of VIME.
        :param joint_learning_layers_sizes:  Only the hidden layers, as the input and output depend on the encoder and input vector size. This corresponds to the classification and clustering heads.
        :param n_known_classes: The number of known classes, or the number of output neurons of the classification head.
        :param n_unknown_classes:  The number of unknown classes, or the number of output neurons of the clustering head.
        :param activation_fct: The activation function that is *between* the first and last layers of each network. Choices : ['relu', 'sigmoid', None].
        :param encoder_last_activation_fct: The very last layer of the encoder. Choices : ['relu', 'sigmoid', None].
        :param ssl_last_activation_fct: The very last layer of the feature estimator network. Choices : ['relu', 'sigmoid', None].
        :param joint_last_activation_fct: The very last layer of the classification and clustering networks. Choices : ['relu', 'sigmoid', None].
        :param p_dropout: The probability of dropout. Use p_dropout=0 for no dropout.
        """
        super(TabularNCDModel, self).__init__()

        # ==================== encoder ====================
        encoder_layers = []
        for i in range(1, len(encoder_layers_sizes)):
            if i == (len(encoder_layers_sizes) - 1):
                simple_layer = get_simple_layer(encoder_layers_sizes[i - 1],
                                                encoder_layers_sizes[i],
                                                add_dropout=False, activation_fct=encoder_last_activation_fct)
            else:
                simple_layer = get_simple_layer(encoder_layers_sizes[i - 1],
                                                encoder_layers_sizes[i],
                                                add_dropout=True, p_dropout=p_dropout, activation_fct=activation_fct)
            [encoder_layers.append(layer) for layer in simple_layer]

        self.encoder = nn.Sequential(*encoder_layers)
        # =================================================

        # ====================== SSL ======================
        if len(ssl_layers_sizes) == 0:
            self.mask_vector_estimator = nn.Sequential(*get_simple_layer(encoder_layers_sizes[-1],
                                                                         encoder_layers_sizes[0],
                                                                         add_dropout=False, activation_fct='sigmoid'))
            self.feature_vector_estimator = nn.Sequential(*get_simple_layer(encoder_layers_sizes[-1],
                                                                            encoder_layers_sizes[0],
                                                                            add_dropout=False, activation_fct=ssl_last_activation_fct))
        else:
            mask_vector_estimator_first_layer = get_simple_layer(encoder_layers_sizes[-1],
                                                                 ssl_layers_sizes[0],
                                                                 add_dropout=True, p_dropout=p_dropout, activation_fct=activation_fct)
            feature_vector_estimator_layer = mask_vector_estimator_first_layer.copy()

            ssl_layers = []
            for i in range(1, len(ssl_layers_sizes)):
                simple_layer = get_simple_layer(ssl_layers_sizes[i - 1],
                                                ssl_layers_sizes[i],
                                                add_dropout=True, p_dropout=p_dropout, activation_fct=activation_fct)
                [ssl_layers.append(layer) for layer in simple_layer]

            mask_vector_estimator_layers = mask_vector_estimator_first_layer + ssl_layers + get_simple_layer(ssl_layers_sizes[-1], encoder_layers_sizes[0], add_dropout=False, activation_fct='sigmoid')
            self.mask_vector_estimator = nn.Sequential(*mask_vector_estimator_layers)

            feature_vector_estimator_layers = feature_vector_estimator_layer + ssl_layers + get_simple_layer(ssl_layers_sizes[-1], encoder_layers_sizes[0], add_dropout=False, activation_fct=ssl_last_activation_fct)
            self.feature_vector_estimator = nn.Sequential(*feature_vector_estimator_layers)
        # =================================================

        # ================= Joint learning ================
        if len(joint_learning_layers_sizes) == 0:
            self.classification_head = nn.Sequential(*get_simple_layer(encoder_layers_sizes[-1],
                                                                       n_known_classes,
                                                                       add_dropout=False, activation_fct=joint_last_activation_fct))
            self.clustering_head = nn.Sequential(*get_simple_layer(encoder_layers_sizes[-1],
                                                                   n_unknown_classes,
                                                                   add_dropout=False, activation_fct=joint_last_activation_fct))
        else:
            classification_head_first_layer = get_simple_layer(encoder_layers_sizes[-1],
                                                               joint_learning_layers_sizes[0],
                                                               add_dropout=True, p_dropout=p_dropout, activation_fct=activation_fct)
            clustering_head_first_layer = classification_head_first_layer.copy()

            joint_learning_layers = []
            for i in range(1, len(joint_learning_layers_sizes)):
                simple_layer = get_simple_layer(joint_learning_layers_sizes[i - 1],
                                                joint_learning_layers_sizes[i],
                                                add_dropout=True, p_dropout=p_dropout, activation_fct=activation_fct)
                [joint_learning_layers.append(layer) for layer in simple_layer]

            classification_head_layers = classification_head_first_layer + joint_learning_layers + get_simple_layer(joint_learning_layers_sizes[-1], n_known_classes, add_dropout=False, activation_fct=joint_last_activation_fct)
            self.classification_head = nn.Sequential(*classification_head_layers)

            clustering_head_layers = clustering_head_first_layer + joint_learning_layers + get_simple_layer(joint_learning_layers_sizes[-1], n_unknown_classes, add_dropout=False, activation_fct=joint_last_activation_fct)
            self.clustering_head = nn.Sequential(*clustering_head_layers)
        # =================================================

    def encoder_forward(self, x):
        return self.encoder(x)

    def classification_head_forward(self, encoded_x):
        return self.classification_head(encoded_x)

    def clustering_head_forward(self, encoded_x):
        return self.clustering_head(encoded_x)

    def vime_forward(self, x):
        encoded_x = self.encoder(x)
        mask_pred = self.mask_vector_estimator(encoded_x)
        feature_pred = self.feature_vector_estimator(encoded_x)
        return mask_pred, feature_pred


def get_simple_layer(size_in, size_out, add_dropout=True, p_dropout=0.3, activation_fct='sigmoid'):
    """
    General function to define a layer with a single dense layer, followed *optionally* by a dropout and activation layer.
    :param size_in: The input size of the dense layer.
    :param size_out:  The output size of the dense layer.
    :param add_dropout: Add a dropout layer of not.
    :param p_dropout: The probability of the dropout layer.
    :param activation_fct: The activation function. Choices : ['relu', 'sigmoid', None].
    :return: List : The layers.
    """
    simple_layer = [nn.Linear(size_in, size_out)]

    if add_dropout is True:
        simple_layer.append(nn.Dropout(p=p_dropout))

    if activation_fct == 'relu':
        simple_layer.append(nn.ReLU())
    elif activation_fct == 'sigmoid':
        simple_layer.append(nn.Sigmoid())
    elif activation_fct == 'tanh':
        simple_layer.append(nn.Tanh())

    return simple_layer
