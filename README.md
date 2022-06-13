# TabularNCD : An adaptation of AutoNovel for tabular data

-----
## How the method works

The datasets are split in this manner:

![alt text](./figures/illustration_novel_tabular_ncd.png)

Some classes will be set as "unlabelled" and the model won't be able to train using their labels (they will only be used for evaluation during training).

Statistical information of the datasets evaluated in the ICDM 2022 paper can be found below:

![alt text](./figures/infos_datasets.png)

The training process is illustrated below:

![alt text](./figures/ncl_model.png)

In step 1, the encoder is pre-trained using all the available data with self-supervised learning
This process is done with the VIME method, which corrupts some features of the inputs.
The model is then trained to predict (1) the mask of the corrupted features (i.e. which features were corrupted) and (2) reconstruct the actual values of the inputs.

![alt text](./figures/ncl_model_2.png)

ToDo : Describe the rest of the training process here...

**Note:** The ranges of values explored in the hyper-parameters Bayesian search are found at the end of this document.

-----
## Note : Virtualenv creation and packages installation
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
    * encoder_layers_sizes : Todo
    * ssl_layers_sizes : Todo
    * joint_learning_layers_sizes : Todo
    * activation_fct : Todo
    * encoder_last_activation_fct : Todo
    * ssl_last_activation_fct : Todo
    * joint_last_activation_fct : Todo
    * dropout : Todo
    * p_m : Todo
    * alpha : Todo
    * batch_size : Batch size of the joint learning step.
    * cosine_topk : Todo
    * M : Size of the memory queue for the data augmentation method.
    * epochs : Number of joint training epochs.
    * transform_method : The variation of the SMOTE-NC insipired method. The \'old\' versions cannot handle categorical features.
    * k_neighbors : The number of neighbors to consider in the data augmentation method.
    * w1 : The classification network trade-off parameter.
    * w2 : The clustering network trade-off parameter.
    * pseudo_labels_method : Todo
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
    ├── figures                        <- 
    └── src                            <- 
        ├── import_utils.py            <- 
        ├── loss_functions.py          <- 
        ├── ncl_memory_module.py       <- 
        ├── TabularNCDModel.py         <- 
        ├── training_procedures.py     <- 
        ├── transforms.py              <- 
        └── utils.py                   <- 

-----
## Explored hyper-parameters values ranges

ToDo
