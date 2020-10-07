"""Preprocess the model predictions"""
from src.evaluation.aux.compute_precision_measures import *
from pathlib import Path
import numpy as np

class SetData():
    def __init__(self, data_set_name, pool_size, pool_setting, budgets, num_reals, eval_window_size, resultsdir, num_reals_tuning, grid_size, load_hyperparameters, hyperparameter_bounds, which_methods, constants):
        """
        Base class to set data for the experiments.

        Parameters:
        :param dataset: Options include {'emotion_detection', 'cifar10_5592', 'cifar10_4070', 'domain_drift'}
        :param pool_size: Size of streaming instances in a realization
        :param pool_setting: It must be 'floating' by default. Options include {'floating', 'fixed'}
        :param budgets: A list of budgets which the model selection methods will be evaluated
        :param num_reals: Number of realizations over which the evaluations are averaged. Set it to a few thousands at least
        :param eval_window: Size of sliding window over which winner frequencies will be measured. Set it to 3 by default
        :param num_reals_tuning: Number of realizations over which the hyperparameters will be tuned
        :param grid_size:  Size of grid which will be used to create a training data set to learn the mapping between the hyperparameters and number of labels each method queries throughout the streaming instances

        Attributes:
        :num_instances:
        :pool_setting:
        :eval_window:
        :num_classes:
        :methods_fullname:
        :methods:
        :num_models:
        :num_reals:
        :budgets:
        :resultsdir:
        :data_set_name:
        :size_entire_pool:
        :num_reals_tuning:
        :grid_size:
        """

        # Attribution to the self
        if data_set_name == 'emotion_detection':
            # Data path
            path_emotiondata = Path(r'resources/datasets/emotion_detection/')

            # Preprocess
            predictions = np.load(str(path_emotiondata) + "/predictions.npy")
            oracle = np.load(str(path_emotiondata) + "/oracle.npy")

            # Dataset specific attributes
            self._predictions = predictions
            self._oracle = oracle
            self._num_classes = 4
            self._num_models = np.size(predictions, 1)
            self._size_entire_pool = np.size(predictions, 0)

        elif data_set_name == 'imagenet':
            # Load and preprocess data
            path_domain = Path(r'resources/datasets/imagenet/')

            # Preprocess data
            predictions = np.load(str(path_domain) + "/predictions.npy")
            predictions -= 1 # Correct the predicted labels to be in between 0 - C-1
            predictions[predictions==-1] = 1001 # Set background labels to C+1
            # Remove the identical models
            idx_range = list(np.arange(np.size(predictions, 1)))
            # Delete the identical models
            del idx_range[92]
            del idx_range[55]
            del idx_range[22]
            predictions = predictions[:, idx_range]
            # process oracle
            oracle = np.load(str(path_domain) + "/oracle.npy")
            oracle -= 1 # Correct the true labels to be in between 0 - C-1
            precs = compute_precisions(predictions, oracle, np.size(predictions, 1))
            # Dataset specific attributes
            self._predictions = predictions
            self._oracle = oracle
            self._num_classes = 1000
            self._num_models = np.size(predictions, 1)
            self._size_entire_pool = np.size(predictions, 0)

        elif data_set_name == 'cifar10_5592':
            # Load and preprocess data
            path_cifar10data5592 = Path(r'resources/datasets/cifar10_5592/')

            # Preprocess data
            predictions = np.load(str(path_cifar10data5592) + "/predictions.npy")
            oracle = np.load(str(path_cifar10data5592) + "/oracle.npy")

            # Dataset specific attributes
            self._predictions = predictions
            self._oracle = oracle
            self._num_classes = 10
            self._num_models = np.size(predictions, 1)
            self._size_entire_pool = np.size(predictions, 0)
            
        elif data_set_name == 'cifar10_4070':
            # Load and preprocess data
            path_cifar10data4070 = Path(r'resources/datasets/cifar10_4070/')

            # Preprocess data
            predictions = np.load(str(path_cifar10data4070) + "/predictions.npy")
            oracle = np.load(str(path_cifar10data4070) + "/oracle.npy")

            # Dataset specific attributes
            self._predictions = predictions
            self._oracle = oracle
            self._num_classes = 10
            self._num_models = np.size(predictions, 1)
            self._size_entire_pool = np.size(predictions, 0)

        elif data_set_name == 'domain_drift':
            # Load and preprocess data
            path_domain = Path(r'resources/datasets/domain_drift/')

            # Preprocess data
            predictions = np.load(str(path_domain) + "/predictions.npy")
            predictions -= 1
            oracle = np.load(str(path_domain) + "/oracle.npy")
            oracle -= 1

            # Dataset specific attributes
            self._predictions = predictions
            self._oracle = oracle
            self._num_classes = 6
            self._num_models = np.size(predictions, 1)
            self._size_entire_pool = np.size(predictions, 0)

        else:
            assert 'Dataset name has not been specified!'

        # Assign constants
        constant_sqbc = constants[0]
        constant_iwal = constants[1]
        constant_efal = constants[2]


        # Attribute other values to self
        self._budgets = budgets
        self._num_reals = num_reals
        self._num_instances = pool_size # This parameter is more experiment dependent, is different (smaller) than the entire pool size
        self._resultsdir = resultsdir
        self._eval_window = eval_window_size
        self._pool_setting = pool_setting
        self._data_set_name = data_set_name
        self._num_reals_tuning = num_reals_tuning
        self._grid_size = grid_size
        self._which_methods = which_methods
        self._load_hyperparameters = load_hyperparameters
        self._hyperparameter_bounds = hyperparameter_bounds
        self._constant_sqbc = constant_sqbc
        self._constant_iwal = constant_iwal
        self._constant_efal = constant_efal

        # Attribute the set of methods and their full names
        all_methods = list(['mp', 'qbc', 'sqbc', 'rs', 'iwal', 'efal'])
        all_methods_fullname = list(
            ['Model Picker', 'Query by Committee', 'Structural Query by Committee', 'Random Sampling',
             'Importance Weighted Active Learning', 'Efficient Active Learning'])
        methods = []
        methods_fullname = []
        for i in range(len(which_methods)):
            if which_methods[i] == 1:
                methods.append(all_methods[i])
                methods_fullname.append(all_methods_fullname[i])

        self._methods = methods
        self._methods_fullname = methods_fullname


    def save_data(self):
        """
        This function saves the setting details.
        """

        # Extract variables
        num_instances = self._num_instances
        pool_setting = self._pool_setting
        eval_window = self._eval_window
        num_classes = self._num_classes
        methods_fullname = self._methods_fullname
        methods = self._methods
        num_models = self._num_models
        num_reals = self._num_reals
        budgets = self._budgets
        resultsdir = self._resultsdir
        data_set_name = self._data_set_name
        size_entire_pool = self._size_entire_pool
        num_reals_tuning = self._num_reals_tuning
        grid_size = self._grid_size
        load_hyperparameters = self._load_hyperparameters
        hyperparameter_bounds = self._hyperparameter_bounds
        which_methods = self._which_methods
        constant_sqbc = self._constant_sqbc
        constant_iwal = self._constant_iwal
        constant_efal = self._constant_efal

        # Save data
        np.savez(str(resultsdir)+"/data.npz", num_instances=num_instances, pool_setting=pool_setting, eval_window=eval_window, num_classes=num_classes, num_models=num_models, num_reals=num_reals, budgets=budgets, methods=methods, methods_fullname=methods_fullname, data_set_name=data_set_name, size_entire_pool=size_entire_pool, num_reals_tuning=num_reals_tuning, grid_size=grid_size, load_hyperparameters=load_hyperparameters, hyperparameter_bounds=hyperparameter_bounds, which_methods=which_methods, constant_iwal=constant_iwal, constant_sqbc=constant_sqbc, constant_efal=constant_efal)


