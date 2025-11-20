from typing import Any, Sequence, Mapping

import torch
import torch.nn as nn
from ember.trainer import BaseTrainer
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
from semantic_f1_score import semantic_f1_score


class SubjectiveClassifierTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_similarity = self.any_dataset.label_similarity

    def input_batch_args(self, batch):
        return batch["encoding"]

    def batch_labels(self, batch: Sequence[Any] | Mapping[str, Any]):
        """Grabs labels from batch."""
        return batch["label"]

    def batch_ids(self, batch):
        return batch["id"]

    def index_label_set(
        self, labels: torch.Tensor | int | list[int]
    ) -> str | list[str]:
        return self.any_dataset.index_label_set(labels)

    def get_eval_scores_from_batch(self, logits):
        if self.any_dataset.multilabel:
            return logits.sigmoid().cpu().tolist()
        return logits.softmax(dim=-1).cpu().tolist()

    def get_eval_labels_from_batch(self, labels):
        return labels.cpu().tolist()

    def get_eval_preds_from_batch(self, logits):
        if self.any_dataset.multilabel:
            return (logits >= 0.5).int().cpu().tolist()
        return logits.argmax(dim=-1).int().cpu().tolist()

    def calculate_cls_loss(
        self, logits, labels, train, aggregate=True, epoch=None
    ):

        if self.any_dataset.multilabel:
            criterion = nn.BCEWithLogitsLoss(
                reduction="mean" if aggregate else "none"
            )
        else:
            criterion = nn.CrossEntropyLoss(
                reduction="mean" if aggregate else "none"
            )
        if logits.shape[0] == 0:
            if not aggregate:
                return torch.tensor(0.0, device=logits.device).expand(
                    logits.shape[0]
                )
            return torch.tensor(0.0, device=logits.device)

        labels = (
            labels.long() if not self.any_dataset.multilabel else labels.float()
        )

        loss = criterion(logits, labels)
        if not aggregate:
            loss = loss.mean(-1)
        return loss

    def evaluation_metrics(
        self, eval_outs, eval_outs_id, eval_extras, data_loader=None
    ):
        sep = self.any_dataset.id_separator

        annotator_info = {
            _id.split(sep)[1]: {
                "true": [],
                "pred": [],
            }
            for _id in eval_outs_id["ids"]
        }
        for _id, true, pred in zip(
            eval_outs_id["ids"],
            eval_outs_id["gt"] or [None] * len(eval_outs_id["ids"]),
            eval_outs_id["preds"] or [None] * len(eval_outs_id["ids"]),
        ):
            annotator_id = _id.split(sep)[1]
            annotator_info[annotator_id]["true"].append(true)
            annotator_info[annotator_id]["pred"].append(pred)

        results = {}

        for annotator_id, info in annotator_info.items():

            if self.any_dataset.multilabel:
                macro_f1 = f1_score(
                    info["true"], info["pred"], average="macro", zero_division=0
                )
                micro_f1 = f1_score(
                    info["true"], info["pred"], average="micro", zero_division=0
                )
                samples_f1 = f1_score(
                    info["true"],
                    info["pred"],
                    average="samples",
                    zero_division=0,
                )

                js = jaccard_score(
                    info["true"],
                    info["pred"],
                    average="samples",
                    zero_division=1,
                )

                f1_scores = f1_score(
                    info["true"], info["pred"], average=None, zero_division=0
                )

                results[annotator_id] = {
                    "jaccard_score": js,
                    "micro_f1": micro_f1,
                    "macro_f1": macro_f1,
                    "samples_f1": samples_f1,
                } | {
                    f"{clss}_f1": f1
                    for clss, f1 in zip(
                        data_loader.dataset.label_set, f1_scores
                    )
                }

                if self.label_similarity is not None:
                    y_true = [self.index_label_set(t) for t in info["true"]]
                    y_pred = [self.index_label_set(p) for p in info["pred"]]
                    macro_sef1 = semantic_f1_score(
                        y_true,
                        y_pred,
                        self.label_similarity,
                        average="macro",
                    )
                    micro_sef1 = semantic_f1_score(
                        y_true,
                        y_pred,
                        self.label_similarity,
                        average="micro",
                    )

                    samples_sef1 = semantic_f1_score(
                        y_true,
                        y_pred,
                        self.label_similarity,
                        average="samples",
                    )

                    results[annotator_id]["semantic_macro_f1"] = macro_sef1
                    results[annotator_id]["semantic_micro_f1"] = micro_sef1
                    results[annotator_id]["semantic_samples_f1"] = samples_sef1

            else:
                results[annotator_id] = dict(
                    accuracy=accuracy_score(info["true"], info["pred"]),
                    f1=f1_score(
                        info["true"],
                        info["pred"],
                        zero_division=0,
                        average="macro",
                    ),
                )

        return results

    def evaluate(self, *args, **kwargs):
        eval_outs, eval_outs_id = super().evaluate(*args, **kwargs)
        eval_outs |= eval_outs.pop("aggregate", None)
        return eval_outs, eval_outs_id
