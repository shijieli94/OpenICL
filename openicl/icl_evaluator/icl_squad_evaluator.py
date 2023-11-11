"""Squad Evaluator"""

from . import APIEvaluator


class SquadEvaluator(APIEvaluator):
    def __init__(self) -> None:
        super().__init__(metric="squad")

    def score(self, predictions, references):
        p_list = [{"prediction_text": pred.split("\n")[0], "id": str(i)} for i, pred in enumerate(predictions)]
        r_list = [{"answers": {"answer_start": [0], "text": [ref]}, "id": str(i)} for i, ref in enumerate(references)]
        return super().score(predictions=p_list, references=r_list)["f1"]
