# Online Active Model Selection for Pre-trained ML Classifiers

This repository is the official implementation of our work on online active model selection for pre-trained ML classifiers.


## Requirements

We can set up the repo either through pip, pipenv or Docker.

### Pip

```bash
pip install -r requirements.txt
```

### Pipenv

```bash
pipenv install
pipenv shell
```

### Docker

```bash
docker build --tag online-active-model-selection .
docker run --it online-active-model-selection
```

## Usage
To run experiment on a set of collected pre-trained models, run this command:

```buildoutcfg
python3 -m experiments.run_experiment [--dataset {'CIFAR10', 'ImageNet', 'Drift', 'EmoContext', 'CIFAR10 V2'}]
                                      [--stream_size STREAM_SIZE] 
                                      [--stream_setting {'float', 'fixed'}] 
                                      [--budgets BUDGETS]
                                      [--num_reals NUM_REALS] 
                                      [--eval_window EVAL_WINDOW] 
                                      [--num_reals_tuning NUM_REALS_TUNING]
                                      [--grid_size GRID_SIZE] 
                                      [--load_hyperparameters {'true', 'false'}] 
                                      [--hyperparameter_bounds HYPERPARAMETER BOUNDS]
                                      [--which_methods WHICH_METHODS] 
                                      [--constants CONSTANTS]

arguments:
--dataset {'CIFAR10', 'ImageNet', 'Drift', 'EmoContext', 'CIFAR10 V2'}
                                      Dataset over which collected pre-trained models will be ranked.
--stream_size STREAM_SIZE             Size of streaming instances in a single realization (denoted by n in the paper).
--stream_setting {'float', 'fixed'}   If 'floating', streaming instances will be drawn randomly from entire test set at each 
                                      realization. Else, a set of streaming instances of size n will be drawn once 
                                      in the beginning, and will be randomly shuffled at each realization.
--budgets BUDGETS                     Labeling budgets over which the algorithms will be tested.
--num_reals NUM_REALS                 Number of realizations over which experiment results are averaged.
--eval_window EVAL_WINDOW             Window size to measure frequency of models being returned as best. 
                                      Not relevant to the results on the submission.
--num_reals_tuning NUM_REALS_TUNING   Number of realizations over which hyperparameter tuning results are averaged.
--grid_size GRID_SIZE                 Size of grid points over which hyperparameter tuning is performed.
--load_hyperparameters {'true', 'false'}
                                      If 'false', tune hyperparameters. Else load the hyperparameters available for 
                                      stream_size and dataset of interest.
--hyperparameter_bounds HYPERPARAMETER_BOUNDS
                                      If load_hyperparameters is false, set bounds for hyperparameter tuning 
                                      which grids points lie in.
--which_methods WHICH_METHODS         List of integers of size 6 that account for the model selection methods in order: 
                                      Model Picker, Query by Committee [1], Structural Query by Committee [2], Random Sampling, 
                                      Importance Weighted Active Learning [3] and Efficient Active Learning [4, 5]. If corresponding 
                                      integer is set to 0, then that model selection method is not tested.
--constants CONSTANTS                 Constant scalings for sampling probability of some methods. This is fixed and 
                                      required to make sure all methods can query arbitrary number of instances.


optional arguments: 
--cluster {'localhost', 'fargate', None, '<dask-scheduler-address-and-port>'}
                                      Specifies the target cluster where experiments can run. We use dask for parallel 
                                      and distributed computing. If cluster is None, we run in single-process mode. If 
                                      cluster is 'localhost' we create a local dask cluster with multiprocessing. If 
                                      cluster is 'fargate' we create a Fargate cluster on AWS (needs AWS config and 
                                      credentials file under ~/.aws). Otherwise, an address of a dask scheduler node 
                                      can be specified and it will be used to run experiment jobs there.
--aws_workers AWS_WORKERS             If cluster is 'fargate' this controls how many workers are created.
```

## Reproducibility

To reproduce the results in the paper run this command:

```buildoutcfg
python3 -m experiments.reproduce_experiment [--dataset {'CIFAR10', 'ImageNet', 'Drift', 'EmoContext', 'CIFAR10 V2'}]

arguments:
--dataset {'CIFAR10', 'ImageNet', 'Drift', 'EmoContext', 'CIFAR10 V2'}
                                      Dataset over which pre-trained models are collected.

optional arguments: 
--cluster {'localhost', 'fargate', None, '<dask-scheduler-address-and-port>'}
                                      Specifies the target cluster where experiments can run. We use dask for parallel 
                                      and distributed computing. If cluster is None, we run in single-process mode. If 
                                      cluster is 'localhost' we create a local dask cluster with multiprocessing. If 
                                      cluster is 'fargate' we create a Fargate cluster on AWS (needs AWS config and 
                                      credentials file under ~/.aws). Otherwise, an address of a dask scheduler node 
                                      can be specified and it will be used to run experiment jobs there.
```

> Upon completion of experiments, it will save the all the results at `resources/results/`. By default, number of 
>realizations is set to 500 for all datasets. 
>Each realization on a single core can take from 1 seconds (EmoContext) to 4 minutes (ImageNet) for all model selection methods 
>in total. 
>
>If you would like to get a glimpse of results by reproducing in a limited time, you can run `python3 -m experiments.reproduce_experiment 'EmoContext'`, which will take slightly less than an hour. For other datasets, we recommend setting number of realization (set `num_reals` in `reproduce_experiment`) to a lower number and `cluster` to `localhost` to benefit from parallelization. 

## Results

Below, we summarize our results on the identification probabilities of model selection methods in returning the true best model on each dataset where pre-trained models are collected. These results can be reproduced by following above instructions. The results may fluctuate up to a very small value of the presented values at different runs.
For results on the other performance measures such as _accuracy gap_ and _regret_, we refer to our submitted manuscipt.

![Identification Probabilities](identification_probabilities.png?raw=true "Identification Probabilities")

### Remarks 
* We demonstrate the power of the Model Picker algorithm up to a budget at which Model Picker returns the true best model with high probability. Budget can be set to any number upto the size of streaming instances n.
* If the size of stream is to be changed, hyperparameters need to be retuned accordingly: `load_hyperpameters = 'false'`. 
* Certain methods such as Structural Query by Committee and Efficient Active Learning cannot query arbitrary number of instances. More precisely, their querying capability is upper bounded by some number due to the nature of the algorithms. In such cases, they may not exceed the budget (one can observe this while reproducing the results). Similarly, Model Picker refuses to query labels if it reaches to a certain identification probability. For example, it does not query more than 2 500 instances on ImageNet dataset.
## Experimental pipeline
### Pre-trained Models
Each set of model collections consist of predictions by some machine learning models on a set of test instances, as well as their respective ground truth labels. Below is further details on the model collections:

The pretrained models are located at `/resources/datasets/`, which include `{'emotion_detection', 'cifar10_5592', 'imagenet', 'domain_drift', cifar10_4070'}` that stand for `{'EmoContext', 'CIFAR10', 'ImageNet', 'Drift', 'CIFAR10 V2'}`, respectively. Each folder consists of three files: `predictions.npy`, `oracle.npy` and `overview`. `predictions.npy` contains prediction matrix of size `size_of_test_set x number of models`, and `oracle.npy` contains the ground truth labels, an array of `size_of_test_set`. Finally, `overview` indicates details on the pre-trained models. We further summarize the set of model collections as follows.

* `CIFAR10`: The models are collected on the CIFAR-10 dataset following [8]. The collection consists of class predictions of 10 000 test instances by 80 models, and the ground truth labels for 10 000 test instances. The number of classes is 10, and the model accuracies on the entire test set (of 10 000 instances) varies from 0.55 to 0.92.
* `ImageNet`: The available models at [9] are collected on the ImageNet dataset. The collection consists of class predictions of 50 000 test instances by 102 models, and the ground truth labels for 50 000 instances. The number of classes is 1 000, and the model accuracies on the entire test set (consisting of 50 000 instances) varies from 0.5 to 0.82.
* `Drift`: The models are collected on the gas sensor dataset [6]. The collection consists of 9 models and their class predictions (over 6 classes) on 3 000 test instances as well as their respective ground truth labels. The model accuracies on the entire pool (of 3 000 instances) varies from 0.25 to 0.6.
* `EmoContext`: The models are collected on the emotion detection dataset [7]. The collection consists of 8 models and their class predictions (over 4 classes) on 5 509 test instances as well as their respective ground truth labels. The model accuracies on the entire pool (of 5 509 instances) varies from 0.88 to 0.92.
* `CIFAR10 V2`: The models are collected on the CIFAR-10 dataset following [8]. The collection consists of class predictions of 10 000 test instances by 80 models, and the ground truth labels for 10 000 instances. The number of classes is 10, and the model accuracies on the entire test set (of 10 000 instances) varies from 0.40 to 0.70.
    
### Pipeline structure
* The code for the main experiment `run_experiment.py` is located at `experiments/`
* The model collections for each dataset are located at `resources/datasets/`
* The results are saved at `resources/results/`
* The methods are located at `src/methods/`
* The evaluation pipeline is located at `src/evaluation/`

We now briefly review the pipeline.

#### ```run_experiment```
The main experiment pipeline consists of the following modules:
* `experiments/base/set_data.py`: process the arguments for the model selection pipeline. Given the `args`, create a class called `data`.
* `experiments/base/tune_hyperpar_base.py`: tune the hyperparameters via grid search. Assign the tuned hyperparameters to `data` and saved results as `hyperparameter.npz`.
* `experiments/base/experiments_base.py`: conduct experiments. Run the main experiment given `data`, and saved the results as `experiment_results.npz`.
* `src/evaluation/evaluation_base.py`: base class for evaluations. Evaluate the experiment results.
* `src/plots/publish_evals.py`: print evaluation results.

### References
[1] Ido Dagan and Sean P. Engelson. Committee-based sampling for training probabilistic classifiers. _In Machine Learning Proceedings_, pages 150-157, 1995.

[2] Christopher Tosh and Sanjoy Dasgupta.  Interactive structure learning with structural query-by-committee, _Advances in Neural Information Processing Systems 31_, pages 1121-1131, 2018.

[3] Alina Beygelzimer, Sanjoy Dasgupta, and John Langford. Importance weighted active learning. _Proceedings of the 26th Annual International Conference on Machine Learning_, pages 49-56, 2008.

[4] Alina  Beygelzimer,  Daniel  Hsu,  Nikos  Karampatziakis,  John  Langford,  and  Tong  Zhang. Efficient active learning, _ICML 2011 Workshop on On-line Trading of Exploration and Exploitation_, 2011.

[5] Alina Beygelzimer, Daniel J Hsu, John Langford, and Tong Zhang. Agnostic active learning without constraints. _In Advances in Neural Information Processing Systems 23_, pages 199-207, 2010.

[6] Alexander Vergara, Shankar Vembu, Tuba Ayhan, Margaret A. Ryan, Margie L. Homer and Ramon Huerta, Chemical gas sensor drift compensation using classifier ensembles, _Sensors and Actuators B: Chemical_ doi: 10.1016/j.snb.2012.01.074, 2012. 

[7] SemEval 2019 https://www.humanizing-ai.com/emocontext.html

[8] Pytorch Classification https://github.com/bearpaw/pytorch-classification

[9] Tensorflow Hub https://tfhub.dev/


