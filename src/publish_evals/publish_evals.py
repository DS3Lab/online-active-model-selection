import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.style.use('classic')
plt.style.use('default')
import seaborn as sns
import numpy as np
sns.set()

"""This function plots the evaluation results for the streaming setting."""

def publish_evals(resultsdir):

    """
    :param resultsdir:
    :return:
    """

    """Load experiments"""
    #    experiment_results = np.load(str(resultsdir) + '/experiment_results.npz')

    """Extract params, vals"""
    # Load data specific details
    data = np.load(str(resultsdir) + "/data.npz")
    methods = data['methods']
    methods_fullname = data['methods_fullname']
    num_instances = data['num_instances']
    num_models = data['num_models']
    num_reals = data['num_reals']
    eval_window = data['eval_window']
    methods = data['methods']
    methods = methods_fullname
    budget = data['budgets']


    """Load evaluations"""
    eval_results = np.load(str(resultsdir) + '/eval_results.npz')

    """Extract evaluations"""
    prob_succ = eval_results['prob_succ']
    acc = eval_results['acc']
    prob_succ_frequent = eval_results['prob_succ_frequent']
    acc_frequent = eval_results['acc_frequent']
    regret = eval_results['regret']
    regret_time = eval_results['regret_time']
    #
    num_queries = eval_results['num_queries']
    log_acc = eval_results['log_acc']
    log_acc_frequent = eval_results['log_acc_frequent']
    true_acc = eval_results['true_acc']
    #
    num_queries_t = eval_results['num_queries_t']
    print(num_queries_t.shape)

    # Determine for which budget point to monitor regret over the stream
    idx_regret = int(round(len(budget)/2))


    """Compute expected and worst-case accuracy gaps:"""
    # Compute the gaps per realization, per budget and per method
    log_gap = np.zeros(log_acc.shape)
    log_gap_frequent = np.zeros(log_gap.shape)
    for i in range(np.size(log_acc, 2)):
        log_gap[:, :, i] = true_acc - np.squeeze(log_acc[:, :, i])
        log_gap_frequent[:, :, i] = true_acc - np. squeeze(log_acc_frequent[:, :, i])
    # Compute the expected accuracy gap
    mean_acc_gap = 100 * np.mean(log_gap, axis=1) # percentage
    worst_acc_gap = 100 * np.max(log_gap, axis=1) # percentage

    mean_acc_gap_frequent = 100 * np.mean(log_gap_frequent, axis=1) # percentage
    worst_acc_gap_frequent = 100 * np.max(log_gap_frequent, axis=1) # percentage



    """Print the evaluation results."""
    for i in np.arange(np.size(log_acc, 2)):
        print('\nMethod: ' + str(methods_fullname[i]) + '  \n|| Number of Queries: ' + str(
            num_queries[i, :]) + '   \n|| Budget: ' + str(budget) + ' \n|| Confidence: ' + str(prob_succ[:, i]) +' \n|| Confidence (frequent): ' + str(prob_succ_frequent[:, i]) +
              ' \n|| Expected accuracy gap: ' + str(mean_acc_gap[:, i]) + '\n|| Worst case accuracy gap: ' + str(
            worst_acc_gap[:, i])+' \n|| Expected accuracy gap (frequent): ' + str(mean_acc_gap_frequent[:, i]) + '\n|| Worst case accuracy gap (frequent): ' + str(
            worst_acc_gap_frequent[:, i]) + ' \n|| Final regret: ' + str(regret[:, i]))

        # Note: If you would like to monitor regret (over stream), please uncomment below. We omit this to avoid printing a huge matrix
        # print('\nMethod: '+str(methods_fullname[i]) + ' \n|| Regret over time: ' +str(regret_time[idx_regret, :, i]))


    return [regret_time, num_queries_t, prob_succ, budget]
