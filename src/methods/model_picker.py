import numpy as np
"""This code runs stream based model picker (proposed algorithm)."""

def model_picker(data, idx_budget, streaming_data_indices, tuning_par, mode):
    """
    :param data:
    :param streaming_data_indices:
    :param tuning_par:
    :param mode: modes include {predictive}
    :return:
    """

    # Set params
    eta_0 = 2 * np.sqrt(np.log(data._num_models))
    if idx_budget == 'tuning mode':
        budget = data._num_instances
    else:
        budget = data._budgets[idx_budget]

    # Edit the input data accordingly with the indices of streaming data
    predictions = data._predictions[streaming_data_indices, :]
    oracle = data._oracle[streaming_data_indices]


    # Initialize
    loss_t = np.zeros(data._num_models) # loss per models
    z_t_log = np.zeros(data._num_instances, dtype=int) # binary query decision
    z_t_budget = np.zeros(data._num_instances, dtype=int) # binary query decision
    posterior_t_log = np.zeros((data._num_instances, data._num_models)) # posterior log
    mp_oracle = np.zeros(data._num_instances)
    hidden_loss_log = np.zeros(data._num_instances, dtype=int)
    It_log = np.zeros(data._num_instances, dtype=int)
    posterior_t = np.ones(data._num_models)/data._num_models

    # For each streaming data instance
    for t in np.arange(1, data._num_instances+1, 1):


        # Edit eta
        eta = eta_0 / np.sqrt(t)


        posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
        # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        posterior_t  /= np.sum(posterior_t)  # normalize

        # Log posterior_t
        posterior_t_log[t-1, :] = posterior_t


        # Compute u_t
        u_t = _compute_u_t(data, posterior_t, eta, predictions[t-1, :], tuning_par, mode)

        # Sanity checks for sampling probability
        if u_t > 1:
            u_t = 1
        if np.logical_and(u_t>=0, u_t<=1):
            u_t = u_t
        else:
            u_t = 0

        # If u_t is in the region of disagreement, don't query anything
        if u_t == 0:
            z_t = 0
            z_t_log[t-1] = z_t
        else:
            # Else, make a random query decision
            z_t = np.random.binomial(size=1, n=1, p=u_t)
            z_t_log[t-1] = z_t

        if z_t == 1:

        # Update loss_t
            loss_t += (np.array((predictions[t-1, :] != oracle[t-1]) * 1) / (u_t + eta/2))
            loss_t = loss_t.reshape(data._num_models, 1)
            loss_t = np.squeeze(np.asarray(loss_t))

        m_star = np.random.choice(list(range(data._num_models)), p=posterior_t)
        # Incur hidden loss
        hidden_loss_log[t-1] = (predictions[t-1, m_star] != oracle[t-1]) * 1
        # print(z_t)
        # print(loss_t)

        # Terminate if it exceeds the budget
        if np.sum(z_t_log) < budget:
            z_t_budget[t-1] = z_t_log[t-1]


    # Labelling decisions as 0's and 1's
    labelled_instances = z_t_log
    ct_log = np.ones(data._num_instances, dtype=int)


    return (labelled_instances, ct_log, z_t_budget, hidden_loss_log, posterior_t_log)


##

def _compute_u_t(data, posterior_t, eta, predictions_c, tuning_par, mode):

    # Compute the coefficients
    coef1 = 8 / eta
    coef2 = 8 / (eta ** 2)

    # Initialize possible u_t's
    u_t_list = np.zeros(data._num_classes)

    # Repeat for each class
    for c in range(data._num_classes):
        # Compute the loss of models if the label of the streamed data is "c"
        loss_c = np.array(predictions_c != c)*1
        #
        # Compute the respective u_t value (conditioned on class c)
        term1 = np.inner(posterior_t, loss_c)
        if mode == 'KL divergence':
            term2 = np.log(np.inner(posterior_t, np.exp(-eta * loss_c)))
            u_t_list[c] = coef1 * term1 + coef2 * term2
        else:
            u_t_list[c] = term1*(1-term1)

    # Return the final u_t
    u_t = tuning_par * np.max(u_t_list)

    return u_t