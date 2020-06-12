import numpy as np
import numpy.matlib

def importance_weighted_active_learning(data, idx_budget, streaming_data_indices, tuning_par, constant_iwal):


    # Set vals, params
    if idx_budget == 'tuning mode':
        budget = data._num_instances
    else:
        budget = data._budgets[idx_budget]

    # Edit the input data accordingly with the indices of streaming data
    predictions = data._predictions[streaming_data_indices, :]
    oracle = data._oracle[streaming_data_indices]

    # Initialize
    p_t_log = np.zeros(data._num_instances) # probability of being queried for the streaming data
    c_t_log = np.zeros(data._num_instances) # weight of each streaming instance: 1/p
    z_t_log = np.zeros(data._num_instances) # query decision
    z_t_budget = np.zeros(data._num_instances, dtype=int)
    models_t = np.ones(data._num_models, dtype=int) # the ensemble at epoch t: 1 if the model is in the ensembele, 0 otherwise
    L_t_log = np.zeros(data._num_models, dtype=float) # error of models at epoch t


    # For each streaming instance
    for t in np.arange(data._num_instances):

        # Is x_t in the region of disagreement?
        dis_t = len(np.unique(predictions[t, :]))

        # Query decision only if x_t is in the region of disagreement
        if dis_t != 1:

            # Set the rejection threshold
            (p_t, models_t_updated) = _loss_weighting(predictions[t, :], t, data._num_classes, 0.1, L_t_log, models_t)
            # #print('pt='+str(p_t))

            p_t = p_t * tuning_par

            # Update the ensemble
            models_t = models_t_updated

            if p_t > 1:
                p_t = 1
            if p_t < 0:
                p_t = 0

            # Log the rejection threshold/probability of being queried
            p_t_log[t] = p_t


            # Randomly decide whether to query its label or not
            z_t = np.random.binomial(size=1, n=1, p=p_t)
            z_t_log[t] = z_t

            # Log c_t's
            if p_t != 0:
                c_t = 1/p_t
            else:
                c_t = 0
            c_t_log[t] = c_t

            # Update L[t] log
            oracle_replicated = np.matlib.repmat(oracle.reshape(data._num_instances, 1), 1, data._num_models)
            loss_accumulated = np.asarray(predictions[:t+1, :] != oracle_replicated[:t+1, :])*1
            ratio = np.multiply(z_t_log[:t+1], c_t_log[:t+1]).reshape(t+1,1)
            ratio_replicated = np.matlib.repmat(ratio, 1, data._num_models)
            L_t_log = np.mean(np.multiply(loss_accumulated, ratio_replicated), axis=0)
        else:
            z_t_log[t] = 0
            c_t_log[t] = 1

            # Terminate if budget is exceeded
        if np.sum(z_t_log) <= budget:
            z_t_budget[t] = z_t_log[t]

    # Labelling decisions as 0's and 1's
    labelled_instances = z_t_log

    return (labelled_instances, c_t_log, z_t_budget)

#
def _loss_weighting(predictions_t, t, num_classes, delta, L_t_log, models_t):

    # Find the ensemble: the models that have survived so far
    models_t_ind = np.where(models_t.reshape(np.size(models_t), 1) == 1)[0]

    # Find the relative L[t-1]
    L_t = np.min(L_t_log[models_t_ind])

    # Compute delta[t-1]
    num_models_t = len(models_t_ind)
    delta_t = _rejection_threshold(t, num_models_t, delta)

    # Compute the upper bound for ensemble learning
    ensemble_threshold = L_t + delta_t

    # Find the hypothesis below the ensemble threshold
    models_t_next = (L_t_log <= ensemble_threshold)

    # Find the overlapping models with already survived ones
    models_t_updated = np.logical_and(models_t_next, models_t)
    num_models = np.size(predictions_t)
    models_t_updated_ind = np.where(models_t_updated.reshape(num_models, 1) == 1)[0]

    # Compute p[t]
    #
    # Initialize the introspective losses
    introspective_losses = np.zeros(num_classes)

    # For each possible label of y_t
    for c in  np.arange(num_classes):
        ###

        # Log the number of models in this epoch
        num_models_t = np.size(models_t_updated_ind)

        # Compute the loss of models.
        loss_models = np.asarray(predictions_t[models_t_updated_ind] != c) * 1

        # Compute the introspective loss.
        introspective_losses[c] = np.max(loss_models) - np.min(loss_models)

    # Set p_t the maximum among all possible pairwise losses
    p_t = np.max(introspective_losses)

    # Check if p_t is outside of [0, 1]
    if p_t > 1:
        p_t = 1

    # Return p_t
    return (p_t, models_t_updated)



def _rejection_threshold(t, num_models_t, delta):

    # Set delta[t] to 0 if no instance has streamed before the current one yet
    if t == 0:
        delta_t = 0

    else:
        t +=1
        # Compute delta_t
        delta = 0.01
        term1 = 8/t
        term2 = np.log(2*t*(t+1)*num_models_t**2 / delta)
        delta_t = np.sqrt(term1*term2)


    return delta_t