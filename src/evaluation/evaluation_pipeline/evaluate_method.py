"""This function evaluates the average performance of a single method over all realizations."""
from src.evaluation.evaluation_pipeline.evaluate_realizations import *
from src.evaluation.aux.compute_precision_measures import *
from numpy import matlib
import numpy as np

from dask.distributed import Client, as_completed
# from dask.bag import Bag, from_sequence
import dask.bag as bg

from tqdm.auto import tqdm, trange

import cloudpickle, zlib

def evaluate_method(data, streaming_instance_log, zt_method_allreals, ct_method_allreals, method, posterior_t_log, client=None):
    """

    :param data:
    :param streaming_instance_log:
    :param zt_method_allreals:
    :param ct_method_allreals:
    :param method:
    :return:
    """

    zt_method_allreals = np.squeeze(np.asarray(zt_method_allreals))
    ct_method_allreals = np.squeeze(np.asarray(ct_method_allreals))

    # Set params
    num_reals = data._num_reals # number of realizations which the evaluation will be averaged over
    freq_window_size = data._eval_window # the window which model frequencies will be calculated over
    num_instances = data._num_instances  # number of instances per realization

    # Initialize all evaluations to zero
    prob_succ_method = 0
    acc_method = 0
    post_ratio_method = 0
    regret_method = 0
    #
    freq_models_method = 0
    gap_star_freqs_method = 0
    gap_freqs_method = 0

    predictions = data._predictions
    oracle = data._oracle

    result = []

    desc = "Realizations (Method: %s)" % method

    if client is None:

        # For each realization for the method of interest, evaluate the realization and add accumulate the results (normalized by number of realizations)
        for i in trange(num_reals, desc=desc):
            result.append(evaluate_realizations((streaming_instance_log[:, i], zt_method_allreals[:, i],
                                                ct_method_allreals[:, i], posterior_t_log[:, :, i]),
                                                predictions, oracle, freq_window_size, method))

    else:

        tqdm.write("Broadcasting data to workers.")
        [predictions_future] = client.scatter([zlib.compress(cloudpickle.dumps(predictions))], broadcast=True)
        [oracle_future] = client.scatter([zlib.compress(cloudpickle.dumps(oracle))], broadcast=True)

        arg_futures = client.scatter([
            (streaming_instance_log[:, i], zt_method_allreals[:, i], ct_method_allreals[:, i], posterior_t_log[:, :, i])
            for i in range(num_reals)]
        )

        tqdm.write("Submitting tasks.")
        result_futures = []
        for i in range(num_reals):
            result_futures.append(client.submit(evaluate_realizations, arg_futures[i], predictions_future, oracle_future, freq_window_size, method, pure=False))
        
        for future in tqdm(as_completed(result_futures), total=num_reals, desc=desc):
            result.append(future.result())

    (true_acc_method, log_acc_method, prob_succ_real, regret_real, post_ratio_real,
    freq_models_method, gap_star_freqs_method, gap_freqs_method, regret_t_method, num_queries_t_method,
    frequent_winner_method, frequent_prob_succ_method, frequent_acc_method, ) = zip(*result)


    acc_method = np.mean(log_acc_method)
    prob_succ_method = np.mean(prob_succ_real)
    regret_method = np.mean(regret_real)
    post_ratio_method = np.mean(post_ratio_real)
    freq_models_method = np.mean(freq_models_real, axis=1)
    gap_star_freqs_method = np.mean(gap_star_freqs_real)
    gap_freqs_method = np.mean(gap_freqs_real)


    """Form the dictionary"""
    evals_method = {
        'prob_succ_method':prob_succ_method,
        'acc_method':acc_method,
        'regret_method':regret_method,
        'post_ratio_method':post_ratio_method,
         #
        'gap_star_freqs_method':gap_star_freqs_method,
        'gap_freqs_method':gap_freqs_method,
        'freq_models_method':freq_models_method,
         #
        'log_acc_method':log_acc_method,
        'true_acc':true_acc_method
    }

    return evals_method












