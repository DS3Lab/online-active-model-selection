import numpy as np

def load_results(data, idx_budget):
    """
    This function loads the experiment results in the results folder for a given budget
    """
    # Load data
    experiment_results = np.load(str(data._resultsdir) + '/experiment_results_'+ 'budget'+str(data._budgets[idx_budget]) + '.npz')

    # Extract vars
    idx_log = experiment_results['idx_log']
    idx_budget_log = experiment_results['idx_budget_log']
    ct_log = experiment_results['ct_log']
    streaming_instances_log = experiment_results['streaming_instances_log']
    hidden_loss_log = experiment_results['hidden_loss_log']
    posterior_log = experiment_results['posterior_log']

    return (idx_log, ct_log, streaming_instances_log, idx_budget_log, posterior_log)
