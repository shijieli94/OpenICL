"""BLEU evaluator"""

from . import APIEvaluator


class BleuEvaluator(APIEvaluator):
    def __init__(self) -> None:
        super().__init__(metric="sacrebleu")
