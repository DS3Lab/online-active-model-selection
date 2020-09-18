from src.evaluation.evaluation_pipeline.evaluate_method import *
from src.evaluation.aux.load_results import *

from dask.distributed import Client, as_completed
from tqdm.auto import tqdm, trange
import cloudpickle, zlib

def evaluate_main(data, client=None):
    """
    This function evaluates the streaming methods one by one, and saves the evaluation results.

    Parameters:
    :param data:

    Returns:

    """

    # Set params
    len_budgets = len(data._budgets)

    # Initialization
    num_queries = np.zeros((len(data._methods), len_budgets))

    # Set params
    num_reals = data._num_reals # number of realizations which the evaluation will be averaged over
    freq_window_size = data._eval_window # the window which model frequencies will be calculated over
    num_instances = data._num_instances  # number of instances per realization
    predictions = data._predictions
    oracle = data._oracle

    # Initialize the evaluations
    prob_succ = np.zeros((len_budgets, len(data._methods)))
    acc = np.zeros((len_budgets, len(data._methods)))
    regret = np.zeros((len_budgets, len(data._methods)))
    post_ratio = np.zeros((len_budgets, len(data._methods)))
    #
    freq_models = np.zeros((len_budgets, data._num_models, len(data._methods)))
    gap_star_freqs = np.zeros((len_budgets, len(data._methods)))
    gap_freqs = np.zeros((len_budgets, len(data._methods)))
    #
    # Initialize the log accuracies
    log_acc = np.zeros((len_budgets, num_reals, len(data._methods)))
    true_acc = np.zeros((len_budgets, num_reals))
    # Regret over time
    regret_time = np.zeros((len_budgets, num_instances, len(data._methods)))
    num_queries_t = np.zeros((len_budgets, num_instances, len(data._methods)))

    if client is not None:

        tqdm.write("Broadcasting model and oracle predictions to workers.")
        [predictions_future] = client.scatter([predictions], broadcast=True)
        [oracle_future] = client.scatter([oracle], broadcast=True)

    # For each budget, repeat the experiment
    for idx_budget in trange(len_budgets, desc="Evaluating Budgets"):
    
        desc = "Evaluating Methods (Budget: %d)" % data._budgets[idx_budget]

        # Load results
        (idx_all, ct_all, streaming_instances_log, idx_queries, posterior_t_log) = load_results(data, idx_budget)
        # idx_budgeted_queries = idx_queries
        # idx_queries = idx_all


        if client is not None:
            tqdm.write("Broadcasting experiment logs to workers.")
            arg_futures = [None for _ in range(len(data._methods))]
            for i in range(len(data._methods)):
                arg_futures[i] = client.scatter([
                    (streaming_instances_log[:, j], idx_queries[:, j, i], ct_all[:, j, i], posterior_t_log[:, :, j])
                    for j in range(num_reals)]
                )
            
            tqdm.write("Submitting tasks.")
            method_result_futures = []
            for i in range(len(data._methods)):
                realization_result_futures = []
                for j in range(num_reals):
                    realization_result_futures.append(client.submit(evaluate_realizations, arg_futures[i][j], predictions_future, oracle_future, freq_window_size, data._methods[i], pure=False, priority=-i))
                
                method_result_futures.append(client.submit(
                    lambda realizations: (
                        np.array(realizations)[:, 0],
                        np.array(realizations)[:, 1],
                        np.array(realizations)[:, 1:].mean(axis=0).tolist()
                    ),
                    realization_result_futures))

        if client is None:

            # Evaluate each method
            for i in trange(len(data._methods), desc=desc):

                method_result = []

                # For each realization for the method of interest, evaluate the realization and add accumulate the results (normalized by number of realizations)
                for j in trange(num_reals, desc="Realizations (Method: %s)" % data._methods[i]):
                    method_result.append(evaluate_realizations((streaming_instances_log[:, j], idx_queries[:, j, i],
                                                        ct_all[:, j, i], posterior_t_log[:, :, j]),
                                                        predictions, oracle, freq_window_size, data._methods[i]))

                (true_acc_method, log_acc_method, prob_succ_real, regret_real, post_ratio_real,
                freq_models_real, gap_star_freqs_real, gap_freqs_real, regret_t, num_queries_t_real,) = zip(*method_result)

                # Raw x-axis
                prob_succ[idx_budget, i] = np.mean(prob_succ_real)
                acc[idx_budget, i] = np.mean(log_acc_method)
                regret[idx_budget, i] = np.mean(regret_real)
                post_ratio[idx_budget, i] = np.mean(post_ratio_real)
                #
                gap_star_freqs[idx_budget, i] = np.mean(gap_star_freqs_real)
                gap_freqs[idx_budget, i] = np.mean(gap_freqs_real)
                freq_models[idx_budget, :, i] = np.mean(freq_models_real)
                # print('freq_models_real:'+str(np.size(freq_models_real)))

                #
                log_acc[idx_budget, :, i] = log_acc_method
                #
                true_acc[idx_budget, :] = true_acc_method

                # Calculate the plain budget usage
                num_queries[i, idx_budget] = np.sum(idx_all[:, :, i]) / data._num_reals
                # print(regret_t)
                print(np.size(regret_t))
                regret_time[idx_budget, :, i] = np.mean(regret_t, axis=0)
                num_queries_t[idx_budget, :, i] = np.mean(num_queries_t_real, axis=0)

        else:

            for i, future in enumerate(tqdm(method_result_futures, total=len(data._methods), desc=desc)):

                true_acc_method, log_acc_method, mean_values = future.result()

                log_acc_method_m, prob_succ_real_m, regret_real_m, post_ratio_real_m, freq_models_real_m, gap_star_freqs_real_m, gap_freqs_real_m, regret_t, num_queries_t_real = mean_values


                # Raw x-axis
                prob_succ[idx_budget, i] = prob_succ_real_m
                acc[idx_budget, i] = log_acc_method_m
                regret[idx_budget, i] = regret_real_m
                post_ratio[idx_budget, i] = post_ratio_real_m
                #
                gap_star_freqs[idx_budget, i] = gap_star_freqs_real_m
                gap_freqs[idx_budget, i] = gap_freqs_real_m
                freq_models[idx_budget, :, i] = freq_models_real_m
                # print('freq_models_real_m:'+str(np.size(freq_models_real_m)))
                #
                log_acc[idx_budget, :, i] = log_acc_method
                true_acc[idx_budget, :] = true_acc_method
                #
                regret_time[idx_budget, :, i] = regret_t
                num_queries_t[idx_budget, :, i] = num_queries_t_real


                # Calculate the plain budget usage
                num_queries[i, idx_budget] = np.sum(idx_all[:, :, i]) / data._num_reals





    """Save evaluations"""
    np.savez(str(data._resultsdir) + '/eval_results.npz',
             prob_succ=prob_succ, acc=acc, regret=regret, post_ratio=post_ratio,
             gap_star_freqs=gap_star_freqs,
             gap_freqs=gap_freqs,
             freq_models=freq_models,
             #
             num_queries=num_queries,
             #
             log_acc=log_acc,
             true_acc=true_acc,
             idx_queries = idx_queries,
             regret_time = regret_time,
             num_queries_t = num_queries_t
             )

    """Form the dictionary"""
    eval_results = {
        'prob_succ':prob_succ,
        'acc':acc,
        'regret':regret,
        'post_ratio':post_ratio,
        #
        'gap_star_freqs':gap_star_freqs,
        'gap_freqs':gap_freqs,
        'freq_models':freq_models,
        #
        'num_queries': num_queries,
        #
        'log_acc':log_acc,
        'true_acc':true_acc,
        #
        'idx_queries':idx_queries,
        #
        'regret_time':regret_time,
        'num_queries_t':num_queries_t
    }

    return eval_results