from pathlib import Path
import numpy as np
from src.evaluation.aux.compute_precision_measures import *
# Load and preprocess data
path_domain = Path(r'resources/datasets/imagenet/')

# Preprocess data
predictions = np.load(str(path_domain) + "/predictions.npy")
predictions -= 1  # Correct the predicted labels to be in between 0 - C-1
predictions[predictions == -1] = 1001  # Set background labels to C+1
oracle = np.load(str(path_domain) + "/oracle.npy")
oracle -= 1
print('Sanity Check:' + str(compute_precisions(predictions, oracle, np.size(predictions, 1))))
prec =compute_precisions(predictions, oracle, np.size(predictions, 1))
print(np.where(prec==0.8444)[0])
print(np.where(prec==0.83466)[0])
print(np.where(prec==0.8406)[0])

