"""ROUGE evaluator"""

from . import APIEvaluator


class RougeEvaluator(APIEvaluator):
    def __init__(self) -> None:
        super().__init__(metric="rouge")
