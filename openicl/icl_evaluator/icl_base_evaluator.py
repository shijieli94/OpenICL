"""Base Evaluator"""


class BaseEvaluator:
    def __init__(self) -> None:
        pass

    def score(self, predictions, references):
        raise NotImplementedError("Method hasn't been implemented yet")
