from experiments.base.set_data import *
from src.evaluation.evaluation_pipeline.evaluate_main import *

from dask.distributed import Client, LocalCluster
from dask_cloudprovider import FargateCluster

if __name__ == '__main__':

    DatasetName = 'imagenet'
    #
    #
    PoolSize = 10000
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
    budgets = [100, 500, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000, 30000, 35000]
    budgets = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 2250, 2500, 3000, 4000, 5000, 7000, 10000]
    # Set other params
    PoolSetting = 'floating'
    WinnerEvalWindow = 15
    grid_size = 250
    num_reals_tuning = 40
    NumReals = 200
    load_hyperparameters = 1

    which_methods = list([1, 1, 1, 1, 0, 0])  # In order, mp, qbc, sqbc, rs, iwal, efal

    hyperparameter_bounds = [hyper_mp, hyper_qbc, hyper_sqbc, hyper_iwal, hyper_efal]  # budget:160max
    hyperparameter_bounds_experiment = []
    for i in np.arange(6):
        if i < 3:
            if which_methods[i] == 1:
                hyperparameter_bounds_experiment.append(hyperparameter_bounds[i])
        elif i > 3:
            if which_methods[i] == 1:
                hyperparameter_bounds_experiment.append(hyperparameter_bounds[i - 1])
    
    """Initiate client"""
    aws_workers=99
    cluster = LocalCluster(processes=True, threads_per_worker=1, memory_limit="4GB", n_workers=40)
    print("Connecting to client.")
    client = Client(address=cluster)
    print("\n >>> Monitoring dashboard: %s \n" % client.dashboard_link)

    """Add results directory."""
    results_dir = "resources/results/imagenet_poolsize10000_numreals200_Date-2020-05-31_Time-22-31-01"

    """Set data."""
    # Set the data
    data = SetData(DatasetName, PoolSize, PoolSetting, budgets, NumReals, WinnerEvalWindow, results_dir, num_reals_tuning,
                   grid_size, load_hyperparameters, hyperparameter_bounds, which_methods, constants)  # data class
    # Save
    data.save_data()

    """Evaluate data."""
    print("\n# Running Final Evaluation\n")
    evaluate_main(data, client=client)
