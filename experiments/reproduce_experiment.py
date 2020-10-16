from experiments.run_experiment import *
from dask.distributed import LocalCluster

def main(dataset_name, cluster=None):

    experiment = dataset_name
    load_hyperparameters = 'true'

    if experiment == 'EmoContext':
        # Emotion Detection
        DatasetName = 'emotion_detection'
        #
        StreamSize = 1000
        #
        hyper_mp = 60
        hyper_qbc = 4
        hyper_sqbc = 4
        hyper_iwal = 5
        hyper_efal = 0.00005
        #
        constants = [150, 0.1, 6]
        #
        budgets = list(np.arange(10, 200, 20))
        #
        grid_size = 250
        num_reals_tuning = 100
        NumReals = 500
        which_methods = list([1, 1, 1, 1, 1, 1])  # In order, mp, qbc, sqbc, rs, iwal, efal

    elif experiment == 'DomainDrift':
        # Emotion Detection
        DatasetName = 'domain_drift'
        #
        StreamSize = 2500
        #
        hyper_mp = 60
        hyper_qbc = 4
        hyper_sqbc = 4
        hyper_iwal = 5
        hyper_efal = 0.00005
        #
        constants = [150, 0.1, 6]
        #
        budgets = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

        #
        grid_size = 250
        num_reals_tuning = 100
        NumReals = 500
        which_methods = list([1, 1, 1, 1, 1, 0])  # In order, mp, qbc, sqbc, rs, iwal, efal
        #
    elif experiment == 'CIFAR10':
        # CIFAR10 55-92
        DatasetName = 'cifar10_5592'
        #
        #
        StreamSize = 5000
        #
        hyper_mp = 2900
        hyper_qbc = 1.47
        hyper_sqbc = 4.54
        hyper_iwal = 0.9
        hyper_efal = 0.00002
        #
        constant_sqbc = 1.4
        constants = [150, 0.1, 5]
        #
        budgets = [50,250,500,750,1000,1250,1500,2000,2500,3000,3500,4000]
        #
        grid_size = 250
        num_reals_tuning = 100
        NumReals = 500
        which_methods = list([1, 1, 1, 1, 1, 1])  # In order, mp, qbc, sqbc, rs, iwal, efal

        #
    elif experiment == 'CIFAR10 V2':
        # CIFAR10 40-70
        DatasetName = 'cifar10_4070'
        #
        StreamSize = 5000
        hyper_mp = 50000
        hyper_qbc = 3
        hyper_sqbc = 10
        hyper_iwal = 1
        hyper_efal = 0.004

        constants = [10, 0.1, 6]  # sqbc: increase
        #
        budgets = [50, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000]
        grid_size = 250
        num_reals_tuning = 100
        NumReals = 500
        which_methods = list([1, 1, 1, 1, 1, 0])  # In order, mp, qbc, sqbc, rs, iwal, efal
        #
    elif experiment == 'ImageNet':
        DatasetName = 'imagenet'
        #
        #
        StreamSize = 10000
        #
        hyper_mp = 135
        hyper_qbc = 22
        hyper_sqbc = 20
        hyper_iwal = 1
        hyper_efal = 0.003
        #
        constant_sqbc = 2
        constants = [constant_sqbc, 0.1, 3]
        #
        budgets = [50, 300, 600, 900, 1200, 1500, 1750, 2000, 2250, 2500, 3000, 4000, 5000, 7000, 10000]
        #
        grid_size = 250
        num_reals_tuning = 100
        NumReals = 500
        which_methods = list([1, 1, 1, 1, 1, 0])  # In order, mp, qbc, sqbc, rs, iwal, efal

    else:
        raise ValueError('The model collection does not exist')



    """Set common params."""
    StreamSetting = 'floating'
    WinnerEvalWindow = 15
    #
    hyperparameter_bounds = [hyper_mp, hyper_qbc, hyper_sqbc, hyper_iwal, hyper_efal]  # budget:160max
    hyperparameter_bounds_experiment = []
    for i in np.arange(6):
        if i < 3:
            if which_methods[i] == 1:
                hyperparameter_bounds_experiment.append(hyperparameter_bounds[i])
        elif i > 3:
            if which_methods[i] == 1:
                hyperparameter_bounds_experiment.append(hyperparameter_bounds[i-1])

    """Run experiment."""
    cluster = None
    # cluster = 'localhost'
    aws_workers = 32
    run_experiment(DatasetName, StreamSize, StreamSetting, budgets, NumReals, WinnerEvalWindow, num_reals_tuning, grid_size, load_hyperparameters, hyperparameter_bounds_experiment, which_methods, constants, cluster, aws_workers)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1 and len(args) != 2:
        print("Missing arguments (or too many)")
        print("Usage: python -m experiments.reproduce_experiment [dataset_name] [cluster (optional)]")
        exit(1)
    else:
        if len(args) == 1:
            main(
                args[0]
            )
        else:
            main(
                args[0],
                args[1]
            )
