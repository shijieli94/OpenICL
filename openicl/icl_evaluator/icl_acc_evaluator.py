"""Acc Evaluator"""

from . import APIEvaluator


class AccEvaluator(APIEvaluator):
    def __init__(self) -> None:
        super().__init__(metric="accuracy")

    def score(self, predictions, references):
        mapping_to_int_dict = {label: idx for idx, label in enumerate(set(map(str, references)))}

        for pred in set(map(str, predictions)):
            if pred not in mapping_to_int_dict.keys():
                mapping_to_int_dict[pred] = len(mapping_to_int_dict)

        golds = [mapping_to_int_dict[str(gold)] for gold in references]
        preds = [mapping_to_int_dict[str(pred)] for pred in predictions]

        return super().score(predictions=preds, references=golds)
