from src.evaluation.aux.compute_precision_measures import *
import tqdm
import zlib, cloudpickle

def evaluate_realizations(log_slice, predictions, oracle, freq_window_size, method):
    """
    This function evaluates the method in interest for given realization of the pool/streaming instances

    Parameters:
    :param predictions: predictions on the streaming instances (specific to the realization in interest)
    :param oracle: ground truth for the streaming instances (specific to the realization in interest)
    :param streaming_instances_i: instances that were part of a stream
    :param zt_real: {0, 1} vectors where 1's indicate the instances that are querid
    :param ct_real: Respective importance weights for the zt_real
    :param posterior_real: the posterior of the model picker only
    :param freq_window_size: the sliding window size where frequency of each model showing up is computed
    :param method: the method in interest

    Returns:
    Realization specific evaluations
    prob_succ_real, acc_real, regret_real, post_ratio_real,
         freq_models_real, gap_star_freqs_real, gap_freqs_rea
    """

    streaming_instances_i, zt_real, ct_real, posterior_real = log_slice
    # predictions = cloudpickle.loads(zlib.decompress(predictions))
    # oracle = cloudpickle.loads(zlib.decompress(oracle))


    # Extract predictions from models and oracle for given streaming instances.
    predictions = predictions[streaming_instances_i, :]
    oracle = oracle[streaming_instances_i]

    # Set params
    num_instances, num_models = predictions.shape

    # Extract true predictions.
    true_precisions = compute_precisions(predictions, oracle, num_models)
    true_winner = np.where(np.equal(true_precisions, np.max(true_precisions)))[0]
    winner_randint = np.random.randint(len(true_winner))
    true_winner_random = true_winner[winner_randint]
    true_acc = true_precisions[true_winner_random]

    # Squeeze the unit dimensions of posterior real and streaming instance indices.
    streaming_instances_i = np.squeeze(streaming_instances_i).astype(int)
    posterior_real = np.squeeze(np.asarray(posterior_real))

    # Convert z[t] to indices format
    # labelled_ins = np.squeeze(np.asarray(zt_real.nonzero())) # the indices whose labels are queried
    labelled_ins = np.ravel(np.asarray(zt_real.nonzero())) # the indices whose labels are queried
    num_labelled = np.size(labelled_ins) # number of queries for this realization ~budget in interest
    if num_labelled == 0:
        labelled_ins = 0
        num_labelled = 1

    # Evaluate the methods upon seeing all the streamed instances

    # Compute the weighted loss
    weighted_losses = compute_weighted_loss(predictions[labelled_ins, :], oracle[labelled_ins], ct_real[labelled_ins], num_models)
    weighted_accuracies = compute_weighted_accuracy(predictions[labelled_ins, :], oracle[labelled_ins], ct_real[labelled_ins], num_models)


    # Declare the winners
    if method == 'mp': # If model picker, declare the winner through its posterior
        arg_winners_t = np.where(np.equal(posterior_real[-1, :].reshape(num_models, 1), np.max(posterior_real[-1, :])))[0]
    else: # else, through the weighted losses
        if np.size(weighted_losses) > 1:
            arg_winners_t = np.where(np.equal(weighted_losses.reshape(num_models, 1), np.min(weighted_losses)))[0]  # Winners of the round
        else:
            arg_winners_t = np.ones(num_models)



    # If multi winners, choose randomly
    len_winners = np.size(arg_winners_t)
    if len_winners > 1:
        idx_winner_t = np.random.choice(len_winners, 1)
        winner_t = arg_winners_t[idx_winner_t]
        winner_t = winner_t.astype(int)
    else:
        winner_t = arg_winners_t.astype(int)

    # Probability of success
    if winner_t in true_winner:
        prob_succ_real = 1
    else:
        prob_succ_real = 0
    # prob_succ_real = (winner_t == true_winner).astype(int)

    # Accuracy of the returned model
    acc_real = true_precisions[winner_t]

    # Log posterior
    if method == 'mp': # If MP, use its own posterior
        posterior = posterior_real[-1, :]
    else: # Else, form a posterior from weighted losses
        if np.sum(weighted_accuracies) == 0:
            posterior = np.ones(num_models)/num_models
        else:
            posterior = weighted_accuracies / np.sum(weighted_accuracies)

    if len(np.unique(posterior)) == 1:
        post_ratio_real = 0
    else:
        best_posterior = np.max(posterior)
        second_best_posterior = float((sorted(set(posterior)))[-2])
        if second_best_posterior == 0:
            post_ratio_real= 0
        else:
            post_ratio_real = np.log(best_posterior/second_best_posterior)


    # Regret

    # Initialize
    loss_true = 0
    loss_winner = 0
    regret_real = 0
    sampled_regret_real = 0
    regret_t = np.zeros(num_instances)
    sampled_regret_t = np.zeros(num_instances)
    num_queries_t_real = np.zeros(num_instances)
    # losses_models = np.zeros(num_models)


    # Compute hidden regret at each instance (not only queried!)
    for t in np.arange(num_instances):

        # losses_winners += (predictions[t, :] != oracle[t]).astype(int)
        if t == 0:
            num_queries_t_real[t] = zt_real[t]
        else:
            num_queries_t_real[t] = num_queries_t_real[t-1]+zt_real[t]


        # Set posterior
        if method == 'mp':  # If MP, use its own posterior
            posterior_t = posterior_real[t, :]
            arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]
        else: # else, check the weighted losses
            posterior_t = np.ones(num_models)/num_models
            if num_labelled == 1:
                labelled_instances_t = 0
            else:
                idx_labelled_instances_transient = np.where(labelled_ins.reshape(num_labelled, 1) < t)[0] # find the location of labelled points that are smaller than t
                labelled_instances_t = labelled_ins[idx_labelled_instances_transient] # find all labelled points so far
            weighted_losses_t = compute_loss(predictions[labelled_instances_t, :], oracle[labelled_instances_t], num_models)
            if np.size(labelled_instances_t)>1:
                if np.sum(weighted_losses_t) == 0: # if no true positive yet, set the posterior uniform
                    arg_winners_t = np.arange(num_models)
                else:
                    arg_winners_t = np.where(np.equal(weighted_losses_t.reshape(num_models, 1), np.min(weighted_losses_t)))[0]
            else:
                arg_winners_t = np.arange(num_models)


        # If multi winners, choose randomly
        len_winners = np.size(arg_winners_t)
        if len_winners > 1:
            idx_winner_t = np.random.choice(len_winners, 1)
            winner_t = arg_winners_t[idx_winner_t]
        else:
            winner_t = arg_winners_t



        # Accumulate the error of returned model
        loss_winner = int((predictions[t, int(winner_t)] != oracle[t])*1)
        # Accumulate the error of true winner
        loss_true =  int((predictions[t, int(true_winner_random)] != oracle[t])*1)

        # Sampled regret time
        m_star = np.random.choice(list(range(num_models)), p=posterior_t)
        # Incur hidden loss
        loss_sampled = (predictions[t, m_star] != oracle[t]) * 1


        regret_real += (loss_winner - loss_true)
        sampled_regret_real += (loss_sampled - loss_true)
        # print(regret_real)
        regret_t[t] = regret_real
        sampled_regret_t[t] = sampled_regret_real
        #

    # Return all
    return (true_acc, acc_real, prob_succ_real, regret_real, regret_t, sampled_regret_real, sampled_regret_t, num_queries_t_real)

#

