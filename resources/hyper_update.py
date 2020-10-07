import numpy as np
from pathlib import Path

dataset = 'imagenet'

if dataset == 'EmoContext':
    '''EmoContext'''
    # main repo
    hyper_dir = Path(r'/Users/nezihemervegurel/Desktop/all/GitHub/online-active-model-selection/resources/hyperparameters/emotion_detection/hyperparameters')
    # load
    hyperparameter_data = np.load(str(hyper_dir) + '.npz')
    grids = hyperparameter_data['grids']
    num_labels = hyperparameter_data['num_labels']
    # remove prev mp hypers
    grids[:, 0] = 0
    num_labels[:, 0] = 0
    # upload new hypers
    hyper_new = Path(r'/Users/nezihemervegurel/Desktop/all/GitHub/online-active-model-selection/emotion_detection_streamsize1000_numreals5_Date-2020-10-06_Time-22-20-19_which_methods110000/hyperparameters')
    hyperparameter_data_new = np.load(str(hyper_new) + '.npz')
    grids_new = hyperparameter_data_new['grids']
    num_labels_new = hyperparameter_data_new['num_labels']
    # log the new values:
    grids[:500, 0] = np.squeeze(grids_new[:,0])
    num_labels[:500, 0] = np.squeeze(num_labels_new[:,0])
    # save the new object
    np.savez(str(hyper_dir) + '.npz', grids=grids, num_labels=num_labels)
elif dataset == 'CIFAR10':
    '''CIFAR10'''
    # main repo
    hyper_dir = Path(r'/Users/nezihemervegurel/Desktop/all/GitHub/online-active-model-selection/resources/hyperparameters/cifar10_5592/hyperparameters')
    # load
    hyperparameter_data = np.load(str(hyper_dir) + '.npz')
    grids = hyperparameter_data['grids']
    num_labels = hyperparameter_data['num_labels']
    # remove prev mp hypers
    grids[:, 0] = 0
    num_labels[:, 0] = 0
    # upload new hypers
    hyper_new = Path(r'/Users/nezihemervegurel/Desktop/all/GitHub/online-active-model-selection/cifar10_5592_streamsize5000_numreals5_Date-2020-10-06_Time-22-20-16_which_methods100000/hyperparameters')
    hyperparameter_data_new = np.load(str(hyper_new) + '.npz')
    grids_new = hyperparameter_data_new['grids']
    num_labels_new = hyperparameter_data_new['num_labels']
    # log the new values:
    grids[:500, 0] = np.squeeze(grids_new)
    num_labels[:500, 0] = np.squeeze(num_labels_new)
    # save the new object
    np.savez(str(hyper_dir) + '.npz', grids=grids, num_labels=num_labels)
elif dataset == 'CIFAR10 (worse models)':
    '''CIFAR10 worse models'''
    # main repo
    hyper_dir = Path(r'/Users/nezihemervegurel/Desktop/all/GitHub/online-active-model-selection/resources/hyperparameters/cifar10_4070/hyperparameters')
    # load
    hyperparameter_data = np.load(str(hyper_dir) + '.npz')
    grids = hyperparameter_data['grids']
    num_labels = hyperparameter_data['num_labels']
    # remove prev mp hypers
    grids[:, 0] = 0
    num_labels[:, 0] = 0
    # upload new hypers
    hyper_new = Path(r'/Users/nezihemervegurel/Desktop/all/GitHub/online-active-model-selection/cifar10_4070_streamsize5000_numreals5_Date-2020-10-06_Time-22-20-06_which_methods100000/hyperparameters')
    hyperparameter_data_new = np.load(str(hyper_new) + '.npz')
    grids_new = hyperparameter_data_new['grids']
    num_labels_new = hyperparameter_data_new['num_labels']
    # log the new values:
    grids[:500, 0] = np.squeeze(grids_new)
    num_labels[:500, 0] = np.squeeze(num_labels_new)
    # save the new object
    np.savez(str(hyper_dir) + '.npz', grids=grids, num_labels=num_labels)
elif dataset == 'imagenet':
    # main repo
    hyper_dir = Path(r'/Users/nezihemervegurel/Desktop/all/GitHub/online-active-model-selection/resources/hyperparameters/imagenet/hyperparameters')
    # load
    hyperparameter_data = np.load(str(hyper_dir) + '.npz')
    grids = hyperparameter_data['grids']
    num_labels = hyperparameter_data['num_labels']
    # remove prev mp hypers
    grids[:, 0] = 0
    num_labels[:, 0] = 0
    # upload new hypers
    hyper_new = Path(r'/Users/nezihemervegurel/Desktop/all/GitHub/online-active-model-selection/imagenet_hypers/hyperparameters')
    hyperparameter_data_new = np.load(str(hyper_new) + '.npz')
    grids_new = hyperparameter_data_new['grids']
    num_labels_new = hyperparameter_data_new['num_labels']
    # log the new values:
    grids[:, 0] = np.squeeze(grids_new)
    num_labels[:, 0] = np.squeeze(num_labels_new)
    # save the new object
    np.savez(str(hyper_dir) + '.npz', grids=grids, num_labels=num_labels)

else:
    print('Error')