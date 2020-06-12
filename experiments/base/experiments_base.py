from src.methods.model_picker import *
from src.methods.random_sampling import *
from src.methods.query_by_committee import *
from src.methods.efficient_active_learning import *
from src.methods.random_sampling_disagreement import *
from src.methods.importance_weighted_active_learning import *
from src.methods.structural_query_by_committee import *

from dask.distributed import Client, as_completed
from tqdm.auto import tqdm, trange

import cloudpickle, zlib

def experiments_base(data, client=None, cache=None):
    """
    The base function for the experiments.
    Parameters:
    :param data: Data attributes
    :param num_processes:
    :param chunksize:
    Returns:
    resources/results/resultsdir/experiment_results.npz
    """

    # Set params
    num_reals = data._num_reals # number of realizations over which the results will be averaged over
    if cache is None:
        cache = {}

    # If client was specified, we can already transfer the data to all workers.
    if client is not None:

        tqdm.write("Broadcasting data to workers.")
        [data_future] = client.scatter([data], broadcast=True)

        # We can also submit all the jobs.
        tqdm.write("Submitting tasks.")
        futures = []
        for i in range(len(data._budgets)):
            required_realizations = num_reals - len(cache.get(i, []))
            futures.append([client.submit(run_realization, data_future, i, pure=False, priority=-i) for _ in range(required_realizations)])


    # For each budget, run the experiment (many realizations)
    for i in trange(len(data._budgets), desc="Iterating over Budgets"):

        desc="Realizations (Budget: %d)" % data._budgets[i]

        result = []

        # Check if some grid points were cached.
        result.extend(cache.get(i, []))
        if len(result) > 0:
            tqdm.write("(Budget: %d) Found %d realizations in the cache." % (data._budgets[i], len(result)))

        if len(result) < num_reals:

            if client is None:
                # If no cluster was specified, we do a simple loop over all realizations, using tqdm to track progress.
                required_realizations = num_reals - len(cache.get(i, []))
                for _ in trange(required_realizations, desc=desc):
                    result.append(run_realization(data, i))
                    cache.setdefault(i, []).append(result[-1])
            else:
                # All jobs were submitted so we just collect results as they arrive and append them to the result list.
                for future in tqdm(as_completed(futures[i]), total=len(futures[i]), desc=desc):
                    result.append(future.result())
                    cache.setdefault(i, []).append(result[-1])

        # Assemble results of the experiment.
        idx_log_all, idx_budget_log_all, ct_log_all, streaming_instances_log_all, hidden_loss_log_all, posterior_log_all = zip(
            *result)


        idx_log = np.stack(idx_log_all, axis=1)
        idx_budget_log = np.stack(idx_budget_log_all, axis=1)
        ct_log = np.stack(ct_log_all, axis=1)
        streaming_instances_log = np.stack(streaming_instances_log_all, axis=1)
        hidden_loss_log = np.stack(hidden_loss_log_all, axis=1)
        posterior_log = np.stack(posterior_log_all, axis=2)

        # Prints
        tqdm.write("\nExperiment Measurements: ")
        for j in np.arange(len(data._methods_fullname)):
            dummy = np.asarray(np.squeeze(idx_log[:, :, j]))
            dummy = np.sum(dummy)/num_reals
            tqdm.write("Method: %-10s Budget: %-10d; Number of Queried Instances: %-10d" % (data._methods[j], data._budgets[i], dummy))
        tqdm.write("")

        """Save the results"""
        np.savez(str(data._resultsdir) + '/experiment_results_'+ 'budget'+str(data._budgets[i]) + '.npz', idx_log=idx_log, idx_budget_log=idx_budget_log,
                 ct_log=ct_log, streaming_instances_log=streaming_instances_log, hidden_loss_log=hidden_loss_log,
                 posterior_log=posterior_log)


def run_realization(data, budget_idx):

    # data = cloudpickle.loads(zlib.decompress(data))

    # Set params
    budget = data._budgets[budget_idx]
    hyperparameters = data._hyperparams[budget_idx, :]
    hyperpars_rs = data._hyperpars_rs[budget_idx]
    constant_sqbc = data._constant_sqbc
    constant_iwal = data._constant_iwal
    constant_efal = data._constant_efal


    num_methods = np.sum(np.asarray(data._which_methods))

    # Initialize the Boolean instance logs and the weights c's for this realization.
    idx_log_i = np.zeros((data._num_instances, num_methods))
    idx_budget_log_i = np.zeros((data._num_instances, num_methods))
    ct_log_i = np.zeros((data._num_instances, num_methods))

    """Set the streaming instances"""
    # If the pool is floating, sample which instance will stream uniformly random
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

    """Run the model selection methods"""
    # Input streaming data to the model selection methods

    num_runing_models = 0

    if 'mp' in data._methods:
        # MODEL PICKER
        tuning_par_mp = hyperparameters[num_runing_models]
        (idx_mp, ct_mp, idx_budget_mp, hidden_loss_log_i, posterior_t_log_i) = model_picker(data, budget_idx, streaming_data_instances_real, tuning_par_mp, 'Variance')
        # Logging
        idx_log_i[:, num_runing_models] = idx_mp
        ct_log_i[:, num_runing_models] = ct_mp
        idx_budget_log_i[:, num_runing_models] = idx_budget_mp
        num_runing_models += 1

    if 'qbc' in data._methods:
        # QUERY BY COMMITTEE
        tuning_par_qbc = hyperparameters[num_runing_models]
        (idx_qbc, ct_qbc, idx_budget_qbc) = query_by_committee(data, budget_idx, streaming_data_instances_real, tuning_par_qbc)
        # Logging
        idx_log_i[:, num_runing_models] = idx_qbc
        ct_log_i[:, num_runing_models] = ct_qbc
        idx_budget_log_i[:, num_runing_models] = idx_budget_qbc
        num_runing_models += 1

    if 'sqbc' in data._methods:
        # STRUCTURAL QUERY BY COMMITTEE
        tuning_par_qbc = hyperparameters[num_runing_models]
        (idx_qbc, ct_qbc, idx_budget_qbc) = structural_query_by_committee(data, budget_idx, streaming_data_instances_real, tuning_par_qbc, constant_sqbc)
        # Logging
        idx_log_i[:, num_runing_models] = idx_qbc
        ct_log_i[:, num_runing_models] = ct_qbc
        idx_budget_log_i[:, num_runing_models] = idx_budget_qbc
        num_runing_models += 1


    num_runing_models_hyper = num_runing_models

    if 'rs' in data._methods:
        # RANDOM SAMPLING
        tuning_par_rs = hyperpars_rs
        (idx_rs, ct_rs, idx_budget_rs) = random_sampling_disagreement(data, budget_idx, streaming_data_instances_real, tuning_par_rs)
        # Logging
        idx_log_i[:, num_runing_models] = idx_rs
        ct_log_i[:, num_runing_models] = ct_rs
        idx_budget_log_i[:, num_runing_models] = idx_budget_rs
        num_runing_models += 1

    if 'iwal' in data._methods:
        # IMPORTANCE WEIGHTED ACTIVE LEARNING
        tuning_par_iwal = hyperparameters[num_runing_models_hyper]
        (idx_iwal, ct_iwal, idx_budget_iwal) = importance_weighted_active_learning(data, budget_idx, streaming_data_instances_real, tuning_par_iwal, constant_iwal)
        # Logging
        idx_log_i[:, num_runing_models] = idx_iwal
        ct_log_i[:, num_runing_models] = ct_iwal
        idx_budget_log_i[:, num_runing_models] = idx_budget_iwal
        num_runing_models += 1
        num_runing_models_hyper += 1

    if 'efal' in data._methods:
        # EFFICIENT ACTIVE LEARNING
        c0 = hyperparameters[num_runing_models_hyper]  # threshold on the efal, increasing means use of more labelling budget
        (idx_efal, ct_efal, idx_budget_efal) = efficient_active_learning(data, budget_idx, streaming_data_instances_real, c0, constant_efal)
        # Logging
        idx_log_i[:, num_runing_models] = idx_efal
        ct_log_i[:, num_runing_models] = ct_efal
        idx_budget_log_i[:, num_runing_models] = idx_budget_efal

    return idx_log_i, idx_budget_log_i, ct_log_i, streaming_data_instances_real, hidden_loss_log_i, posterior_t_log_i
