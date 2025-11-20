import os
import random
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig, AutoConfig
from ember.trainer import BaseTrainer
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
from legm import from_namespace

from sef1_validation.base_datasets import TextDataset, TokenizationMixin
from sef1_validation.benchmarks import (
    SemEval2018Task1Ec,
    GoEmotions,
    MFRC,
    Hatexplain,
    MSPPodcast,
)


class FilterBERT(nn.Module):
    """BERT-based model for filtering labels.

    Attributes:
        bert: large LM with a BertModel-like 'interface'.
        classifier: FFN on top of CLS.
    """

    @staticmethod
    def argparse_args():
        return dict(
            model_name_or_path=dict(
                type=str,
                help="model to load into BERT parts of model",
            ),
            dropout_prob=dict(
                default=0.1,
                type=float,
                help="dropout before final linear layer",
            ),
        )

    def __init__(
        self,
        config: PretrainedConfig,
        dropout_prob: float = 0.1,
    ):
        """Init.

        Args:
            config: LM configuration from `AutoConfig`.
            dropout_prob: dropout before final linear layer.
        """
        super().__init__()

        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob

        try:
            self.bert = AutoModel.from_config(config, add_pooling_layer=False)
        except TypeError:
            self.bert = AutoModel.from_config(config)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size, 1),
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_lm: str, *args, **kwargs
    ) -> "FilterBERT":
        config = AutoConfig.from_pretrained(pretrained_lm)
        model = cls(config, *args, **kwargs)
        try:
            model.bert.load_state_dict(
                AutoModel.from_pretrained(
                    pretrained_lm, add_pooling_layer=False
                ).state_dict()
            )
        except TypeError:
            model.bert.load_state_dict(
                AutoModel.from_pretrained(pretrained_lm).state_dict()
            )
        return model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward propagation. Classification based on CLS token.

        Args:
            `transformers`-style LM inputs.

        Returns:
            Logits whose number is equal to `len(self.class_inds)`,
            contextual representation used for each is average
            of outputs specified by each list.
        """

        last_hidden_state = self.bert(*args, **kwargs).last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        preds = self.classifier(cls_token).squeeze(-1)
        return preds


class FilterDatasetforTransformers(TokenizationMixin, TextDataset):
    @staticmethod
    def argparse_args():
        return TokenizationMixin.argparse_args() | TextDataset.argparse_args()

    annotator_labels = False

    def __init__(self, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs, for_llm=False)
        self.original_label_set = self.label_set
        self.label_set = ["unreasonable", "reasonable"]
        # only annotations["aggregate"] has the debug_len ids
        usable_ids = list(self.annotations["aggregate"])

        random.seed(seed)

        for k in list(self.annotations["aggregate"]):
            labels = self.annotations["aggregate"][k]
            labels = self.index_label_set(
                labels, alternative_label_set=self.original_label_set
            )
            text = self.examples[k]["text"]
            self.examples[k]["text"] = (", ".join(labels), text)
            rnd_id = random.sample(usable_ids, k=1)[0]
            rnd_labels = self.annotations["aggregate"][rnd_id]

            new_k = k + "_rnd"
            # add it here for _extend_dataset_for_annotators
            self.annotations["aggregate"][new_k] = rnd_labels

            rnd_labels = self.index_label_set(
                rnd_labels, alternative_label_set=self.original_label_set
            )

            self.examples[new_k] = {
                "text": (", ".join(rnd_labels), text),
                "original_text": self.examples[k]["original_text"],
            }

        self.real_ids = list(self.examples.keys())
        self.ids, self.annotator2inds = self._extend_dataset_for_annotators()

    def _extend_dataset_for_annotators(self):
        ids = []
        annotator2inds = {}

        for worker_id, annotator_data in self.annotations.items():
            if (
                worker_id == "aggregate" and self.annotation_mode == "annotator"
            ) or (
                worker_id != "aggregate" and self.annotation_mode == "aggregate"
            ):
                continue
            for example_id in annotator_data:
                annotator2inds.setdefault(worker_id, []).append(len(ids))
                ids.append((example_id, worker_id))

        return ids, annotator2inds

    def __getitem__(self, idx):
        example_id, worker_id = self.ids[idx]
        data = deepcopy(self.examples[example_id])
        data["label"] = torch.tensor(int(not example_id.endswith("_rnd")))
        return dict(id=example_id + self.id_separator + worker_id, **data)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])

        return batch

    def index_label_set(
        self,
        label: torch.Tensor | int | list[int],
        alternative_label_set: list[str] | None = None,
    ) -> str | list[str]:
        """Returns label names given numerical `label`."""

        sampling_label_set = (
            alternative_label_set if alternative_label_set else self.label_set
        )

        if not self.multilabel and alternative_label_set is None:
            if torch.is_tensor(label):
                label = int(label.item())
            if isinstance(label, list):
                assert len(label) == 1
                label = int(label[0])
            return sampling_label_set[int(label)]

        if torch.is_tensor(label):
            label = label.tolist()
        return [sampling_label_set[i] for i, l in enumerate(label) if l == 1]


class FilterBERTSemEval2018Task1EcDataset(
    FilterDatasetforTransformers, SemEval2018Task1Ec
):

    multilabel = False

    @staticmethod
    def argparse_args():
        return (
            FilterDatasetforTransformers.argparse_args()
            | SemEval2018Task1Ec.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FilterBERTGoEmotionsDataset(FilterDatasetforTransformers, GoEmotions):

    multilabel = False

    @staticmethod
    def argparse_args():
        return (
            FilterDatasetforTransformers.argparse_args()
            | GoEmotions.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FilterBERTMFRCDataset(FilterDatasetforTransformers, MFRC):

    multilabel = False

    @staticmethod
    def argparse_args():
        return (
            FilterDatasetforTransformers.argparse_args() | MFRC.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FilterBERTHatexplainDataset(FilterDatasetforTransformers, Hatexplain):
    @staticmethod
    def argparse_args():
        args = (
            FilterDatasetforTransformers.argparse_args()
            | Hatexplain.argparse_args()
        )
        del args["root_dir"]
        return args

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FilterBERTMSPPodcastDataset(FilterDatasetforTransformers, MSPPodcast):
    @staticmethod
    def argparse_args():
        return (
            FilterDatasetforTransformers.argparse_args()
            | MSPPodcast.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


DATASETS = {
    "SemEval": FilterBERTSemEval2018Task1EcDataset,
    "GoEmotions": FilterBERTGoEmotionsDataset,
    "MFRC": FilterBERTMFRCDataset,
    "Hatexplain": FilterBERTHatexplainDataset,
    "MSPPodcast": FilterBERTMSPPodcastDataset,
}


class FilterBERTTrainer(BaseTrainer):

    def input_batch_args(self, batch):
        return batch["encoding"]

    def batch_labels(self, batch):

        if "pred_label" in batch:
            batch["label"][batch["label"] != batch["pred_label"]] = -1

        return batch["label"]

    def batch_ids(self, batch):
        return batch["id"]

    def calculate_cls_loss(
        self, logits, labels, train, aggregate=True, epoch=None
    ):
        criterion = nn.BCEWithLogitsLoss(
            reduction="mean" if aggregate else "none"
        )
        labels = labels.float()
        loss = criterion(logits, labels)
        return loss

    def get_eval_scores_from_batch(self, logits):
        return logits.sigmoid().cpu().tolist()

    def get_eval_labels_from_batch(self, labels):
        return labels.cpu().tolist()

    def get_eval_preds_from_batch(self, logits):
        return (logits >= 0.5).int().cpu().tolist()

    def evaluation_metrics(
        self, eval_outs, eval_outs_id, eval_extras, data_loader=None
    ):
        sep = self.any_dataset.id_separator

        annotator_info = {
            _id.split(sep)[1]: {
                "pred_1": [],
                "pred_0": [],
                "pred": [],
                "true": [],
            }
            for _id in eval_outs_id["ids"]
        }
        for _id, true, pred in zip(
            eval_outs_id["ids"],
            eval_outs_id["gt"] or [None] * len(eval_outs_id["ids"]),
            eval_outs_id["preds"] or [None] * len(eval_outs_id["ids"]),
        ):
            annotator_id = _id.split(sep)[1]
            if true == 1:
                annotator_info[annotator_id]["pred_1"].append(pred)
            elif true == 0:
                annotator_info[annotator_id]["pred_0"].append(pred)
            annotator_info[annotator_id]["pred"].append(pred)
            annotator_info[annotator_id]["true"].append(true)

        results = {}

        for annotator_id, info in annotator_info.items():
            results[annotator_id] = dict(
                rand_pair_accuracy=accuracy_score(
                    info["pred_0"], [1] * len(info["pred_0"])
                ),
                gold_pair_accuracy=accuracy_score(
                    info["pred_1"], [1] * len(info["pred_1"])
                ),
                f1=f1_score(
                    info["true"],
                    info["pred"],
                    average="macro",
                    zero_division=0,
                ),
                accuracy=accuracy_score(info["true"], info["pred"]),
            )

        return results

    def evaluate(self, *args, **kwargs):
        eval_outs, eval_outs_id = super().evaluate(*args, **kwargs)
        eval_outs |= eval_outs.pop("aggregate", None)
        return eval_outs, eval_outs_id
