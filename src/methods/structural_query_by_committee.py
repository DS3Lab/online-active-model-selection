import numpy as np
import scipy.stats as stats

def structural_query_by_committee(data, idx_budget, streaming_data_indices, tuning_par, constant_sqbc):


    # Set vals, params
    if idx_budget == 'tuning mode':
        budget = data._num_instances
    else:
        budget = data._budgets[idx_budget]

    # Edit the input data accordingly with the indices of streaming data
    predictions = data._predictions[streaming_data_indices, :]
    oracle = data._oracle[streaming_data_indices]

    # Initialize
    prior = np.ones(data._num_models) / data._num_models
    posterior = prior
    z_t_log = np.zeros(data._num_instances, dtype=int)
    z_t_budget = np.zeros(data._num_instances, dtype=int)
    loss_acc = np.zeros(data._num_models)
    loss_t = 0

    # If the strategy is adaptive,
    for t in range(data._num_instances):

        if len(np.unique(predictions[t, :])) != 1: # If the instance is in the region of disagreement
            # Randomly sample two models from the posterior
            #posterior[posterior<0.01] = 0.01
            posterior = posterior / np.sum(posterior)
            g1, g2 = np.random.choice(data._num_models, p=posterior, size=2, replace=True)

            disagreement = (predictions[:t+1, g1] != predictions[:t+1, g2]).astype(int)
            p_t = np.mean(disagreement) * constant_sqbc
            if p_t > 1:
                p_t = 1
            if np.logical_and(p_t>=0, p_t<=1):
                p_t = p_t
            else:
                p_t = 0
            z_t = np.random.binomial(size=1, n=1, p=p_t)

            # If queried, update the loss
            if z_t == 1:
                loss_t = (predictions[t, :] != oracle[t]).astype(int)
                # Accumulate the loss
                loss_acc += loss_t

                # Update posterior
                beta = tuning_par
                exp_loss_t = np.exp(-beta * loss_t)
                posterior = np.multiply(posterior, exp_loss_t)
                posterior = posterior / np.sum(posterior) # normalize posterior
        else:
            z_t = 0


        z_t_log[t] = z_t


        # Terminate if budget is exceeded
        if np.sum(z_t_log) <= budget:
            z_t_budget[t] = z_t_log[t]

    # Labelling decisions as 0's and 1's
    labelled_instances = z_t_log
    ct_log = np.ones(data._num_instances, dtype=int)


    return (labelled_instances, ct_log, z_t_budget)