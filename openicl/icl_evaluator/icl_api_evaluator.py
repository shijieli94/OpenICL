"""API evaluator"""

from . import BaseEvaluator


class APIEvaluator(BaseEvaluator):
    def __init__(self, metric) -> None:
        super().__init__()

        import evaluate

        self.metric = evaluate.load(metric)

    def score(self, predictions, references):
        assert len(predictions) == len(references)
        scores = self.metric.compute(predictions=predictions, references=references)
        return scores
