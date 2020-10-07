from experiments.base.tune_hyperpar_base import *
from experiments.base.experiments_base import *
from src.evaluation.evaluate_base import *
from experiments.base.set_data import *
from src.publish_evals.publish_evals import *
from datetime import datetime
import time
import os
import shelve
import sys
from dask.distributed import Client, LocalCluster
from dask_cloudprovider import FargateCluster

def run_experiment(dataset, stream_size, stream_setting, budgets, num_reals, eval_window, num_reals_tuning, grid_size, load_hyperparameters, hyperparameter_bounds, which_methods, constants, cluster=None, aws_workers=32):
    """
    The main script for running online model selection experiments.
    """

    start_time = time.time()

    # Initialize dask cluster if it was specified. Otherwise, we will not use any parallelism.
    if cluster is not None:

        if cluster == "localhost":
            cluster = LocalCluster(processes=True, threads_per_worker=1, memory_limit="4GB", n_workers=8)
        elif cluster == "fargate":
            cluster = FargateCluster(n_workers=aws_workers, image="<here>",
                                     worker_extra_args=["--nprocs", "4", "--nthreads", "1"],
                                     scheduler_timeout="55 minutes")
            raise ValueError("Please upload repo image to dockerhub and paste link <here>.")

        print("Connecting to client.")
        client = Client(address=cluster)
        print("\n >>> Monitoring dashboard: %s \n" % client.dashboard_link)

    else:
        print("No cluster specified. Running in single-process mode.")
        client = None

    """Create a results directory."""
    now = datetime.now().strftime("_Date-%Y-%m-%d_Time-%H-%M-%S") # get datetime
    which_methods_print = ''.join(map(str, which_methods))
    os.mkdir('resources/results/'+dataset+'_streamsize'+str(stream_size)+'_numreals'+str(num_reals)+str(now)+'_which_methods'+str(which_methods_print)) # create the folder
    results_dir = Path('resources/results/'+dataset+'_streamsize'+str(stream_size)+'_numreals'+str(num_reals)+str(now)+'_which_methods'+str(which_methods_print)) # assign it to the results directory var

    """Set data."""
    # Set the data
    data = SetData(dataset, stream_size, stream_setting, budgets, num_reals, eval_window, results_dir, num_reals_tuning, grid_size, load_hyperparameters, hyperparameter_bounds, which_methods, constants) # data class
    # Save
    data.save_data()

    # Load the cache.
    # with shelve.open("run_experiment.cache") as cachedb:
    #
    """Hyperparameter tuning."""
    print("\n# Tuning Hyperparameters\n")
    hyperparams = tune_hyperpar_base(data, client=client, cache=None)
    data._hyperparams = hyperparams # assign hyperpars to the data

    """Run experiments."""
    print("\n# Running Experiments\n")
    experiments_base(data, client=client, cache=None)

    #
    #
    """Evaluate data."""
    print("\n# Running Final Evaluation\n")
    Evals(data, client=client)

    """Announce the evaluation results."""
    publish_evals(results_dir)

    elapsed_time = time.time() - start_time

    print("\n# Work Completed...\n")
    print("Elapsed time: %s" % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print("The experiment results can be found at: %s" % str(results_dir))

    # Close the client connection if it was opened.
    if client is not None:
        client.close()
        print("Client closed.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 12:
        print("Missing arguments (or too many)")
        print("Usage: python -m experiments.run_experiments [dataset_name] [stream_size] [stream_setting] [budgets] [num_reals] [winner_eval_window] [num_reals_tuning] [grid_size] [load_hyperparameters (true/false)] [cluster]")
        exit(1)
    else:

        if args[8] == "true":
            load_hyperparameters = True
        elif args[8] == "false":
            load_hyperparameters = None
        else:
            raise ValueError("Incorrect value for load hyperparameters")

        if load_hyperparameters:
            hyperparameter_bounds = list(map(lambda x: float(x), args[9].split(",")))
        else:
            if args[9] != "empty":
                raise ValueError("Load hyper parameters must be empty")
            hyperparameter_bounds = []

        if len(args) == 12:
            run_experiment(
                args[0],  # dataset,
                int(args[1]),  # pool_size,
                args[2],  # pool_setting,
                list(map(lambda x: int(x), args[3].split(","))),  # budgets,
                int(args[4]),  # num_reals,
                int(args[5]),  # eval_window,
                int(args[6]),  # num_reals_tuning,
                int(args[7]),  # grid_size,
                load_hyperparameters,
                hyperparameter_bounds,  # hyperparameter_bounds,
                list(map(lambda x: int(x), args[10].split(","))),  # which_methods
                float(args[11]),  # constant_sqbc,
            )

        if len(args) == 13:
            cluster = args[12]
            run_experiment(
                args[0],  # dataset,
                int(args[1]),  # pool_size,
                args[2],  # pool_setting,
                list(map(lambda x: int(x), args[3].split(","))),  # budgets,
                int(args[4]),  # num_reals,
                int(args[5]),  # eval_window,
                int(args[6]),  # num_reals_tuning,
                int(args[7]),  # grid_size,
                load_hyperparameters,
                hyperparameter_bounds,  # hyperparameter_bounds,
                list(map(lambda x: int(x), args[10].split(","))),  # which_methods
                float(args[11]),  # constant_sqbc,
                cluster
            )

        if len(args) == 14:
            cluster = args[12]
            aws_workers = int(args[13])

            run_experiment(
                args[0],  # dataset,
                int(args[1]),  # pool_size,
                args[2],  # pool_setting,
                list(map(lambda x: int(x), args[3].split(","))),  # budgets,
                int(args[4]),  # num_reals,
                int(args[5]),  # eval_window,
                int(args[6]),  # num_reals_tuning,
                int(args[7]),  # grid_size,
                load_hyperparameters,
                hyperparameter_bounds,  # hyperparameter_bounds,
                list(map(lambda x: int(x), args[10].split(","))),  # which_methods
                float(args[11]),  # constant_sqbc,
                cluster,
                aws_workers
            )

        if len(args) > 14:
            raise ValueError("Too many arguments")

