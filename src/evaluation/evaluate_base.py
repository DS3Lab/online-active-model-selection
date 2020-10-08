"""Base class for the evaluations."""
from src.evaluation.evaluation_pipeline.evaluate_main import *

class Evals:

    def __init__(self, data, client=None):

        """Evaluate methods"""
        eval_results = evaluate_main(data, client=client)

        """Assigns evaluations to the self"""
        self._prob_succ = eval_results['prob_succ']
        self._acc = eval_results['acc']
        self._regret = eval_results['regret']
        self._sampled_regret = eval_results['sampled_regret']
        self._num_queries = eval_results['num_queries']
        self._log_acc = eval_results['log_acc']
        self._true_acc = eval_results['true_acc']
        self._num_queries_t = eval_results['num_queries_t']
        self._regret_time = eval_results['regret_time']
        self._sampled_regret_time = eval_results['sampled_regret_time']




