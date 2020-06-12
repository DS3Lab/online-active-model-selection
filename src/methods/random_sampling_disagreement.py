import numpy as np
"""Random sampling code for stream based model selection."""
def random_sampling_disagreement(data, idx_budget, streaming_data_indices, tuning_par_rs):
    """
    :param data:
    :param streaming_data_indices:
    :return:
    """

    # Set params
    num_instances = data._num_instances
    budget = data._budgets[idx_budget]

    # Initialize
    z_t_log = np.zeros(num_instances,  dtype=int)
    z_t_budget = np.zeros(num_instances)
    ct_log = np.ones(data._num_instances, dtype=int)

    # Set probability of querying specific to the given budget
    p_budget = tuning_par_rs

    # Identify the instances in the region of disagreement
    predictions_sample = data._predictions[streaming_data_indices, :]
    loc_disagreement, num_disagreement = measure_disagreement(predictions_sample)
    idx_disagreement = np.squeeze(np.asarray(np.nonzero(loc_disagreement))).astype(int)

    # Randomly select queries
    z_temp = np.random.binomial(1, p=p_budget, size=num_disagreement)
    z_t_log[idx_disagreement] += z_temp

    # Set the budgeted indices variables
    for i in np.arange(num_instances):
        if np.sum(z_t_log[:i+1]) <= budget:
            z_t_budget[i] += z_t_log[i]
    # print(np.sum(z_t_log))
    return (z_t_log, ct_log, z_t_budget)



def measure_disagreement(predictions):
    """This function counts the number of instances in the region of disagreement."""

    # Set params
    n, m = predictions.shape

    # Initialize
    loc_disagreement = np.zeros(n)

    # For each instance, count the number of non-unique elements
    for i in np.arange(n):
        num_uniques = len(np.unique(predictions[i, :]))
        if num_uniques != 1: # If models have different predictions, set the respective index to one
            loc_disagreement[i] += 1

    # Count the total number of instances in the region of disagreement
    num_disagreement = np.sum(loc_disagreement).astype(int)

    return loc_disagreement, num_disagreement
