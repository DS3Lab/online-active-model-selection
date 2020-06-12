import numpy as np
import numpy.matlib

def compute_precisions(pred, orac, num_models):
    """
    This function computes the agreements
    """
    # Replicate oracle realization
    orac_rep = np.matlib.repmat(orac.reshape(np.size(orac), 1), 1, num_models)

    # Compute errors
    true_pos = (pred == orac_rep) * 1

    # Compute the weighted loss
    precisions = np.mean(true_pos, axis=0)

    # Squeeze precision
    precisions = np.squeeze(np.asarray(precisions))

    return precisions

"""
This function computes the agreements between two methods
"""

def compute_agreements(pred, orac, num_models):
    """
    This function computes the agreements
    """
    # Replicate oracle realization
    orac_rep = np.matlib.repmat(orac.reshape(np.size(orac), 1), 1, num_models)

    # Compute errors
    true_pos = (pred == orac_rep) * 1

    # Compute the weighted loss
    agreements = np.sum(true_pos, axis=0)

    # Reduce the extra dimension
    agreements = np.squeeze(np.asarray(agreements))

    return agreements



def compute_weighted_loss(pred, orac, ct_method, num_models):

    #ct_method = np.asarray(ct_method)


    """
    This function computes the weighted loss
    """

    # Replicate oracle realization
    orac_rep = np.matlib.repmat(orac.reshape(np.size(orac), 1), 1, num_models)

    # Compute errors
    errors = (pred != orac_rep)*1

    # Replicate the weights
    # print('ct shape:'+str(ct_method.shape))
    # print('orac shape:' + str(orac)[0])
    # print('ctshape = ' + str(np.size(ct_method.shape)))
    # print('oracshape = ' + str(ct_method.shape))

    if ct_method.shape != ():
        ct_method_rep = np.matlib.repmat(ct_method.reshape(np.size(orac), 1), 1, np.size(pred, 1))
        # Compute the weighted errors
        weighted_errors = np.multiply(errors, ct_method_rep)
        # Compute the weighted loss
        weighted_loss = np.mean(weighted_errors, axis=0)
        weighted_loss = np.squeeze(np.asarray(weighted_loss))
    else:
        weighted_loss = 0
    #weighted_loss = np.mean(errors, axis=0)

    return weighted_loss


def compute_weighted_accuracy(pred, orac, ct_method, num_models):

    #ct_method = np.asarray(ct_method)


    """
    This function computes the weighted loss
    """

    # Replicate oracle realization
    orac_rep = np.matlib.repmat(orac.reshape(np.size(orac), 1), 1, num_models)

    # Compute errors
    true_positives = (pred == orac_rep)*1

    # Replicate the weights
    # print('ct shape:'+str(ct_method.shape))
    # print('orac shape:' + str(orac)[0])
    # print('ctshape = ' + str(np.size(ct_method.shape)))
    # print('oracshape = ' + str(ct_method.shape))

    if ct_method.shape != ():
        ct_method_rep = np.matlib.repmat(ct_method.reshape(np.size(orac), 1), 1, np.size(pred, 1))
        # Compute the weighted errors
        weighted_true_positives = np.multiply(true_positives, ct_method_rep)
        # Compute the weighted loss
        weighted_true_positives = np.mean(weighted_true_positives, axis=0)
        weighted_true_positives = np.squeeze(np.asarray(weighted_true_positives))
    else:
        weighted_true_positives = 0
    #weighted_loss = np.mean(errors, axis=0)

    return weighted_true_positives



def compute_loss(pred, orac, num_models):

    """
    This function computes the weighted loss
    """

    # Replicate oracle realization
    orac_rep = np.matlib.repmat(orac.reshape(np.size(orac), 1), 1, num_models)

    # Compute errors
    errors = (pred != orac_rep)*1

    # Compute the weighted loss
    loss = np.mean(errors, axis=0)
    loss = np.squeeze(np.asarray(loss))

    return loss
