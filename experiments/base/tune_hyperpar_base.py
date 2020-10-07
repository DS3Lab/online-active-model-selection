import numpy as np
from src.methods.model_picker import *
from src.methods.random_sampling import *
from src.methods.query_by_committee import *
from src.methods.efficient_active_learning import *
from src.evaluation.aux.compute_precision_measures import *
from src.methods.structural_query_by_committee import *
from pathlib import Path
from src.methods.importance_weighted_active_learning import *
from dask.distributed import Client, as_completed
from tqdm.auto import tqdm, trange
import cloudpickle, zlib

from scipy import optimize

def tune_hyperpar_base(data, client=None, cache=None):

    """
    The base function for the experiments.

    Parameters:
    :param data: attributes to data
    :param num_processes:
    :param chunksize:

    Returns:
    hyperparameters for each budget and for each method (excluding random sampling)
    """

    # Set params
    budgets = data._budgets
    hyperparameter_bounds = data._hyperparameter_bounds
    if cache is None:
        cache = {}

    # Initialize
    which_methods = data._which_methods
    idx_all_methods_exclude_random = np.asarray([0, 1, 2, 4, 5])
    which_methods = np.asarray(which_methods)
    idx_methods_exclude_random = np.asarray(which_methods[idx_all_methods_exclude_random]).nonzero()
    num_models_exclude_random = np.sum(np.asarray(which_methods[idx_all_methods_exclude_random])) # count every method that will run except random sampling
    hyperpars = np.zeros((len(budgets), num_models_exclude_random))


    if data._load_hyperparameters is 'true': # If 'true', load the hyperparameters.

        # Load data
        hyper_dir = Path(r'resources/hyperparameters/'+str(data._data_set_name)+'/hyperparameters')
        hyperparameter_data = np.load(str(hyper_dir) + '.npz')

        # Assign hyperparameters and query degrees
        grids = hyperparameter_data['grids']
        num_labels = hyperparameter_data['num_labels']

        grids = np.squeeze(grids[:, idx_methods_exclude_random])
        num_labels = np.squeeze(num_labels[:, idx_methods_exclude_random])

        hyperpars_rs = np.zeros(len(budgets))


    else: # If not 'true', tune the hyperparameters

        # Set params
        num_reals_tuning = data._num_reals_tuning # number of realizations over which hyperparameters will be tuned

        # Set grids
        # Initialization
        grid_size = data._grid_size
        grids = np.zeros((grid_size, num_models_exclude_random))


        # Note: 4 comes from the size of model selection methods excluding random sampling. It does not require hyperparameter tuning.
        for grid_method in range(num_models_exclude_random):
            min_grid = 0.00000000001
            max_grid = float(hyperparameter_bounds[grid_method])
            if np.logical_or(np.logical_or(which_methods[0]==1, which_methods[2]==1), which_methods[5]==1): # for some methods, place grid points logarithmically
                max_grid = np.log2(max_grid)
                min_grid = -12 # good for cifar10 40-70 for all
                if np.logical_or(np.logical_and(which_methods[0]==1, which_methods[2]==1), which_methods[4]==1):
                    min_grid = -6
                grids[:, grid_method] = np.logspace(min_grid, max_grid, num=grid_size, base=2)
            else:
                grids[:, grid_method] = np.linspace(min_grid, max_grid, num=grid_size)

            hyperpars_rs = np.zeros(len(budgets))


        # Initialize the inputs
        num_labels = np.zeros((grid_size, num_models_exclude_random)) # remove the coordinate of random sampling

        # If client was specified, we can already transfer the data to all workers.
        if client is not None:

            tqdm.write("Broadcasting data to workers.")
            [data_future] = client.scatter([data], broadcast=True)

            # We can also submit all the jobs.
            tqdm.write("Submitting tasks.")
            futures = []
            for i in range(grid_size):
                required_realizations = num_reals_tuning #- len(cache.get(i, []))
                futures.append([client.submit(run_realization, data_future, grids[i, :], pure=False, priority=-i) for _ in range(required_realizations)])

        # Run for each grid point.
        for i in trange(grid_size, desc="Hyperparameter Tuning Grid"):

            tuning_parameters = grids[i, :] # set tuning parameters of methods to a grid coordinate
            desc = "Realizations (Grid point: %d/%d)" % (i+1, grid_size)

            result = []

            # Check if some grid points were cached.
            # result.extend(cache.get(i, []))
            if len(result) > 0:
                tqdm.write("(Grid point: %d/%d) Found %d realizations in the cache." % (i+1, grid_size, len(result)))
            
            if len(result) < num_reals_tuning:
            
                if client is None:
                    # If no cluster was specified, we do a simple loop over all realizations, using tqdm to track progress.
                    required_realizations = num_reals_tuning - len(cache.get(i, []))
                    for _ in trange(required_realizations, desc=desc):
                        result.append(run_realization(data, tuning_parameters))
                        cache.setdefault(i, []).append(result[-1])
                else:
                    # All jobs were submitted so we just collect results as they arrive and append them to the result list.
                    for future in tqdm(as_completed(futures[i]), total=len(futures[i]), desc=desc):
                        result.append(future.result())
                        cache.setdefault(i, []).append(result[-1])

            # Assemble results of the experiment.
            idx_log_all = list(zip(*result))
            idx_log = np.stack(idx_log_all, axis=1)

            # # Calculate the average number of queries
            # tuned_methods = list(range(len(data._methods))) # list coordinate of methods
            # del tuned_methods[3] # remove random sampling

            # For each method, measure the expected number of queries throughout the streaming instances (over all realizations)
            for j in np.arange(num_models_exclude_random):
                num_labels[i, j] = np.sum(idx_log[:, :, j]) / num_reals_tuning

    for method_id in np.arange(num_models_exclude_random):

        # :find hyperparameter for each budget and evaluate (every method except random sampling)
        for i in np.arange(np.size(budgets)):
            if num_models_exclude_random != 1:
                idx_closest = np.argmin(abs(num_labels[:, method_id] - budgets[i]))
                # Assign true closes to the hyperparameters
                hyperpars[i, method_id] = grids[idx_closest, method_id]
            else:
                idx_closest = np.argmin(abs(num_labels - budgets[i]))
                # Assign true closes to the hyperparameters
                hyperpars[i] = grids[idx_closest]

    # Hyperparameter for random sampling
    if which_methods[3] == 1:
        num_disagreements_real = measure_disagreement(data._predictions) * data._num_instances / data._size_entire_pool # resize the disagreements on the entire pool to the streaming instances
        hyperpars_rs = budgets/num_disagreements_real # hyperparameters for the random sampling
        hyperpars_rs[hyperpars_rs>1] = 1 # If the hyperparameter is greater than 1, set it to 1 (they will be probability of querying)

    data._hyperpars_rs = hyperpars_rs
    # Save the results.
    np.savez(str(data._resultsdir) + '/hyperparameters.npz', hyperpars=hyperpars, budgets=data._budgets, grids=grids, num_labels=num_labels, hyperpars_rs=hyperpars_rs)

    # print('Grids= '+str(grids))
    # print(' Number of labels= ' + str(num_labels))

    return hyperpars


def run_realization(data, tuning_parameters):

    # Initialize
    which_methods = data._which_methods
    which_methods = np.asarray(which_methods)
    idx_all_methods_exclude_random = np.asarray([0, 1, 2, 4, 5])
    idx_methods_exclude_random = np.asarray(which_methods[idx_all_methods_exclude_random]).nonzero()
    num_models_exclude_random = np.sum(np.asarray(which_methods[idx_all_methods_exclude_random])) # count every method that will run except random sampling

    # data = cloudpickle.loads(zlib.decompress(data))

    # Set the mode of operation
    mode = 'tuning mode'

    # Initialize the query decision for instances
    idx_log_i = np.zeros((data._num_instances, num_models_exclude_random))

    """Set the streaming instances"""
    # If the stream is floating, draw streaming instances uniformly at random
    if data._pool_setting == 'floating':
        # Set the streaming instances for this realization
        streaming_data_instances = np.random.permutation(int(data._size_entire_pool))  # shuffle the entire pool
        streaming_data_instances_real = streaming_data_instances[:data._num_instances]  # select first n instance
    else:
        streaming_data_instances_fixed = np.random.permutation(
            int(data._size_entire_pool))  # shuffle the entire pool
        streaming_data_instances_fixed = streaming_data_instances_fixed[
                                         :data._num_instances]  # select first n instances

        random_perm = np.random.permutation(data._num_instances)  # shuffle the instances
        streaming_data_instances_real = streaming_data_instances_fixed[random_perm]  # update the streaming order
    #



    num_runing_models = 0
    constant_sqbc = data._constant_sqbc
    constant_iwal = data._constant_iwal
    constant_efal = data._constant_efal

    if 'mp' in data._methods:
        # MODEL PICKER
        tuning_par_mp = tuning_parameters[num_runing_models]
        (idx_mp, ct_mp, idx_budget_mp, hidden_loss_log_i, posterior_t_log_i) = model_picker(data, mode, streaming_data_instances_real, tuning_par_mp, 'Variance')
        # Logging
        idx_log_i[:, num_runing_models] = idx_mp
        num_runing_models += 1

    if 'qbc' in data._methods:
        # QUERY BY COMMITTEE
        tuning_par_qbc = tuning_parameters[num_runing_models]
        (idx_qbc, ct_qbc, idx_budget_qbc) = query_by_committee(data, mode, streaming_data_instances_real, tuning_par_qbc)
        # Logging
        idx_log_i[:, num_runing_models] = idx_qbc
        num_runing_models += 1

    if 'sqbc' in data._methods:
        # STRUCTURAL QUERY BY COMMITTEE
        tuning_par_sqbc = tuning_parameters[num_runing_models]
        (idx_sqbc, ct_sqbc, idx_budget_sqbc) = structural_query_by_committee(data, mode, streaming_data_instances_real, tuning_par_sqbc, constant_sqbc)
        # Logging
        idx_log_i[:, num_runing_models] = idx_sqbc
        num_runing_models += 1

    if 'iwal' in data._methods:
        # IMPORTANCE WEIGHTED ACTIVE LEARNING
        tuning_par_iwal = tuning_parameters[num_runing_models]
        (idx_iwal, ct_iwal, idx_budget_iwal) = importance_weighted_active_learning(data, mode, streaming_data_instances_real, tuning_par_iwal, constant_iwal)
        # Logging
        idx_log_i[:, num_runing_models] = idx_iwal
        num_runing_models += 1
    #
    if 'efal' in data._methods:
        # EFFICIENT ACTIVE LEARNING
        c0 = tuning_parameters[num_runing_models]  # threshold on the efal, increasing means use of more labelling budget
        (idx_efal, ct_efal, idx_budget_efal) = efficient_active_learning(data, mode, streaming_data_instances_real, c0, constant_efal)
        # Logging
        idx_log_i[:, num_runing_models] = idx_efal
        num_runing_models += 1

    return idx_log_i


def measure_disagreement(predictions):
    """This function counts the number of instances in the region of disagreement."""

    # Set params
    n, m = predictions.shape

    # Initialize
    idx_disagreement = np.zeros(n)

    # For each instance, count the number of non-unique elements
    for i in np.arange(n):
        num_uniques = len(np.unique(predictions[i, :]))
        if num_uniques != 1: # If models have different predictions, set the respective index to one
            idx_disagreement[i] += 1

    # Count the total number of instances in the region of disagreement
    num_disagreement = np.sum(idx_disagreement)

    return num_disagreement
