import numpy as np
"""Random sampling code for stream based model selection (unused)."""
def random_sampling(data, idx_budget, streaming_data_indices):
    """
    :param data:
    :param streaming_data_indices:
    :return:
    """

    # Set params
    num_instances = data._num_instances
    budget = data._budgets[idx_budget]

    p_budget = budget/num_instances

    # Randomly select queries
    z_t_log = np.random.binomial(1, p=p_budget, size=num_instances)

    # Set other variables
    z_t_budget = z_t_log
    ct_log = np.ones(data._num_instances, dtype=int)

    return (z_t_log, ct_log, z_t_budget)