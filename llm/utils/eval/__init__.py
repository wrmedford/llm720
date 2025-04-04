"""Model evaluation utilities."""

from llm.utils.eval.benchmark import evaluate_model_on_benchmark
from llm.utils.eval.perplexity import evaluate_perplexity
from llm.utils.eval.size import calculate_model_params, analyze_peer_efficiency

__all__ = [
    'evaluate_model_on_benchmark',
    'evaluate_perplexity',
    'calculate_model_params',
    'analyze_peer_efficiency'
]
