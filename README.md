# TabularNCD :  A Method for Discovering Novel Classes in Tabular Data

Table of contents
  * [How the method works](#how-the-method-works)
  * [Virtualenv creation and packages installation](#virtualenv-creation-and-packages-installation)
  * [Script usage example](#script-usage-example)
  * [Detailed usage](#detailed-usage)
  * [Directory structure](#directory-structure)
  * [Explored hyper-parameters values ranges](#explored-hyper-parameters-values-ranges)
  * [Supplementary materials](#supplementary-materials)


-----
## How the method works

The datasets are split in this manner:

![alt text](./figures/illustration_novel_tabular_ncd.png)

Some classes will be set as "unlabelled" and the model won't be able to train using their labels (they will only be used for evaluation during training).

Statistical information of the datasets evaluated in the ICDM 2022 paper can be found below:

![alt text](./figures/infos_datasets.png)

The training process is illustrated below:

![alt text](./figures/ncl_model.png)

In step 1, the goal is to capture a common and informative representation of both x_train and x_unlab.\
Therefore, the encoder is pre-trained using all the available training data with self-supervised learning.\
This process is done with the VIME method, which corrupts some features of the inputs.\
The model is then trained to predict 1) the mask of the corrupted features (i.e. which features were corrupted) and 2) reconstruct the actual values of the inputs.

![alt text](./figures/ncl_model_2.png)

In this step, two new networks are added on top of the previously initialized encoder, each solving different tasks on different data.\
The first is a classification network trained to predict 1) the known classes from x_train with the ground-truth labels and 2) a single class formed of the aggregation of the unlabeled data.\
The second is another classification network trained to  predict the novel classes from x_unlab.


-----
## Virtualenv creation and packages installation
This project was written using **python 3.7.9**. The libraries are described in requirements.txt.

It is recommended to create a virtual environment with *virtualenv* to install the exact versions of the packages used in this project.
You will first need to install *virtualenv* with pip:
> pip install virtualenv

Then create the virtual environment:
> virtualenv my_python_environment

Finally, activate it using :
> source my_python_environment/bin/activate

At this point, you should see the name of your virtual environment in parentheses on your terminal line.

You can now install the required libraries inside your virtual environment with:
> pip install -r requirements.txt


-----
## Script usage example
Display the help:
> python TabularNCD.py -h

Execution with the optimized hyper-parameters:
> python TabularNCD.py --dataset_name Pendigits --hyper_parameters_path auto

**Note :**
If you don't use the optimized hyper-parameters with the 'auto' value (or any other file), 
and that you don't define values, 
the scrip will use the default values and will most likely have poor performance.


-----
## Detailed usage

The parameters of the TabularNCD.py script are:

    * [required] dataset_name : The name of the dataset.
    * use_cuda : Set to True if you want the code to be run on your GPU. If set to False, code will run on CPU.
    * log_lvl : Change the log display level.
    * hyper_parameters_path : Path to the hyper-params file. Set to 'auto' to find it in .\data\dataset_name\hyperparameters.json
    * ssl_lr : Learning rate of the mode in the self-supervised learning phase.
    * lr_classif : Learning rate of the classification network in the joint learning phase.
    * lr_cluster : Learning rate of the clustering network in the joint learning phase.
    * encoder_layers_sizes : The sizes of the encoder's layers. Must include the input and output sizes.
    * ssl_layers_sizes : The hidden layers sizes of the mask and feature vector estimators. Do not include input and output sizes.
    * joint_learning_layers_sizes : The hidden layers sizes of the classification and clustering networks. Do not include input and output sizes.
    * activation_fct : The activation function used in the hidden layers of the encoder. Default = 'relu'. Choices = ['relu', 'sigmoid', 'tanh', None].
    * encoder_last_activation_fct : The activation function of the very last layer of the encoder. Default = None. Choices = ['relu', 'sigmoid', 'tanh', None].
    * ssl_last_activation_fct : The activation function of the very last layer of the feature estimator network. Default = None. Choices = ['relu', 'sigmoid', 'tanh', None].
    * joint_last_activation_fct : The activation function of the very last layer of the classification and clustering networks. Default = None. Choices = ['relu', 'sigmoid', 'tanh', None].
    * dropout : The dropout probability.
    * p_m :  Corruption probability
    * alpha : Loss_vime = mask_estim_loss + alpha * feature_estim_loss.
    * batch_size : Batch size of the joint learning step.
    * cosine_topk : The percentage of the maximum number of pairs in a mini-batch that are considered positive.
    * M : Size of the memory queue for the data augmentation method.
    * epochs : Number of joint training epochs.
    * transform_method : The variation of the SMOTE-NC insipired method. The \'old\' versions cannot handle categorical features.
    * k_neighbors : The number of neighbors to consider in the data augmentation method.
    * w1 : The classification network trade-off parameter.
    * w2 : The clustering network trade-off parameter.
    * pseudo_labels_method : The pseudo labels definition method. Default = 'top_k_cosine_per_instance'. Choices = ['cosine', 'top_k_cosine', 'top_k_cosine_faster', 'top_k_cosine_per_instance', 'ranking'].
    * use_ssl : Use SSL to initialize the encoder or not.
    * freeze_weights : Freeze the weights of the encoder's layer (except the last one) after SSL initialization or not.


-----
## Directory structure
    .
    ├── .gitignore
    ├── README.md                      <- This file
    ├── requirements.txt               <- The required packages
    ├── TabularNCD.py                  <- The main script to launch
    ├── data                           <- The datasets (train & test) in csv, along with the hyper-parameters and links to download
    │   ├── ForestCoverType            <- File is too large, please download and pre-process it before using dataset
    │   ├── HumanActivityRecognition   <- File is too large, please download and pre-process it before using dataset
    │   ├── LetterRecognition          <- Dataset available
    │   ├── mnist                      <- Dataset available
    │   ├── Pendigits                  <- Dataset available
    │   ├── Satimage                   <- Dataset available
    │   └── USCensus1990               <- File is too large, please download and pre-process it before using dataset
    ├── figures                        <- The training metrics curves
    └── src                            <- The source code of the project
        ├── import_utils.py            <- The functions to import the different datasets used here
        ├── loss_functions.py          <- The loss functions used in training
        ├── ncl_memory_module.py       <- A simple class to store the M most recent training instances from the previous batches
        ├── TabularNCDModel.py         <- The TabularNCD model clas
        ├── training_procedures.py     <- The SSL and joint training methods
        ├── transforms.py              <- The data augmentation methods
        └── utils.py                   <- Diverse useful functions


-----
## Explored hyper-parameters values ranges

For all datasets, the same ranges were explored:

  * lr_classif : uniform distribution of float in [0.000001, 0.01]
  * lr_cluster : uniform distribution of float in [0.000001, 0.01]
  * cosine_topk : uniform distribution of float in [1.0, 40.0]
  * k_neighbors : uniform distribution of int in [1, 50]
  * w1 : uniform distribution of float in [0.1, 1.0]
  * w2 : uniform distribution of float in [0.1, 1.0]
  * dropout : uniform distribution of float in [0.0, 0.60]
  * activation_fct : One of [Sigmoid, ReLU, None].


-----
## Supplementary materials

ToDo experiments comparing lambda threshold pseudo labels definition method vs cosine topk...
