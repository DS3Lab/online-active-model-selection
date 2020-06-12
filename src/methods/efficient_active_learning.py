import numpy as np
import sys
import mpmath
sys.modules['sympy.mpmath'] = mpmath
from sympy.solvers.solvers import *

def efficient_active_learning(data, idx_budget, streaming_data_indices, c0, constant_efal):



    # Set vals, params
    c1 = 1
    c2 = c1
    if idx_budget == 'tuning mode':
        budget = data._num_instances
    else:
        budget = data._budgets[idx_budget]

    # Edit the input data accordingly with the indices of streaming data
    predictions = data._predictions[streaming_data_indices, :]
    oracle = data._oracle[streaming_data_indices]


    # Initialize
    p_t_log = np.zeros(data._num_instances)
    z_t_log = np.zeros(data._num_instances)
    z_t_budget = np.zeros(data._num_instances, dtype=int)

    # Repeat for each streaming instance
    for t in range(data._num_instances):

        # if no data streamed in before, set err to 0
        if t == 0:
            err = 0
        else: # Else, compute the error of models
            err = _compute_err(data, predictions[:t, :], oracle[:t], t, z_t_log[:t], p_t_log[:t])


        # Is x_t in the region of disagreement?
        dis_t = len(np.unique(predictions[t, :]))

        # Query decision only if x_t is in the region of disagreement
        if dis_t != 1:

            # Find the errors of best and the second best model
            #
            # The best model
            h_t = np.min(err)
            #
            # The second best model
            if len(np.unique(err)) == 1:
                h_t_ = h_t
            else:
                h_t_ = np.flip(sorted(set(err)))[-2]

            # Compute G[t]
            G_t = h_t_ - h_t

            # Compute the threshold
            if t == 0:
                threshold = 1000
            else:
                threshold = _compute_threshold(t, c0, constant_efal)

            # Compute P[t]
            if G_t <= threshold:
                p_t = 1
            else:
                s = _compute_s(G_t, data._num_models, t, c0, c1, c2)
                p_t = s
            if p_t > 1:
                p_t = 1
            elif p_t < 0:
                p_t = 0
            else:
                p_t = p_t
            # Toss a coin
            z_t = np.random.binomial(size=1, n=1, p=float(p_t))
            # Log the result
        else: # If x_t is not in the region of disagreement, do not query
            p_t = 0
            z_t = 0

        p_t_log[t] += p_t
        z_t_log[t] += z_t

        # Terminate if budget is exceeded
        if np.sum(z_t_log) <= budget:
            z_t_budget[t] = z_t_log[t]

    # Assign z[t]'s to labelled instances
    p_t_log[p_t_log==0] = 1
    c_t_log = np.divide(1, p_t_log)


    return (z_t_log, c_t_log, z_t_budget)


def _compute_err(data, predictions_s, oracle_s, t, z_t_s, p_t_s):

    # Compute the error
    #
    # Compute the loss
    oracle_replicated = np.matlib.repmat(oracle_s.reshape(t, 1), 1, data._num_models)
    loss_s = np.asarray(predictions_s != oracle_replicated)*1
    #
    # Compute the weights
    p_t_s[p_t_s==0] = 1
    ratio = np.divide(z_t_s, p_t_s)
    ratio_replicated = np.matlib.repmat(ratio.reshape(t, 1), 1, data._num_models)
    #
    # Error computed by
    err = np.mean(np.multiply(ratio_replicated, loss_s), axis=0)

    return err
#

def _compute_threshold(t, c0, constant_efal):

    # num_streamed[t] = t+1
    t = t+1

    # Set params
    #c0 = 16 * np.log(num_models * 2 * (3 + t * np.log2(t)) * t * (t+1) / delta) / np.log(t+1)

    # Set terms
    term2 = c0 * np.log(t) / (t - 1)
    term1 = np.sqrt(term2)

    # Compute the threshold
    threshold = term1 + term2

    return threshold*constant_efal
#

def _compute_s(G_t, num_models, t, c0, c1, c2):

    # num_streamed[t] = t+1
    t = t + 1

    # Set terms
    term2 = c0 * np.log(t) / (t - 1)
    term1 = np.sqrt(term2)

    # # Set variable
    # x = Symbol('x')
    #
    # # c1 = 5
    # # c2 = 5
    # Solve the equation
    # s = solve(term1 * (c1/sqrt(x) - c1 + 1) + term2 * (c2/x - c2 + 1) - G_t, x)
    #
    term_1 = 2 * G_t * term2
    term_2 = term2 * np.sqrt(4 * G_t + 1)
    term_3 = 2 * G_t**2

    s = []
    x_1 = (term_1 - term_2 + term2)/term_3
    x_2 = (term_1 + term_2 + term2) / term_3

    s.append(x_1)
    s.append(x_2)

    # Find the ind of positive solution
    s = np.array(s)
    ind_pos = np.where(np.logical_and((s > 0), (s < 1)))[0]

    if len(ind_pos) == 0:
        p_t = 0
    elif len(ind_pos) == 1:
        p_t = s[ind_pos]
    else:
        p_t = np.mean(s)

    p_t = p_t

    return p_t