import os
import langcodes
import json
import re
import csv
from typing import Any, Iterator
from collections import Counter
from pathlib import Path

import torch
import yaml
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from sef1_validation.base_datasets import TextDatasetWithPriors
from sef1_validation.metrics.label_similarity import PLUTCHIK_EMOTION_SIMILARITY


class SemEval2018Task1Ec(TextDatasetWithPriors):
    """Plain text dataset for `SemEval 2018 Task 1: Affect in Tweets`
    (https://competitions.codalab.org/competitions/17751). Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDatasetWithPriors` for attributes.
        language: language to load.
    """

    multilabel = True
    annotator_labels = False
    name = "SemEval 2018 Task 1"
    source_domain = "Twitter"
    label_similarity = PLUTCHIK_EMOTION_SIMILARITY

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = TextDatasetWithPriors.argparse_args()
        args.update(
            dict(
                language=dict(
                    type=str, default="english", help="language to load"
                )
            )
        )
        return args

    def __init__(self, language, *args, **kwargs):
        """Initializes dataset.

        Args:
            language: language to load.
            Check `TextDataset` for other arguments.
        """
        self.language = language
        super().__init__(*args, **kwargs)

    def _load_data(
        self, split: str
    ) -> tuple[list[str], list[str], torch.Tensor, list[str]]:
        split_mapping = dict(
            train="train", dev="dev", test="test-gold", smalldev="smalldev"
        )
        filename = os.path.join(
            self.root_dir,
            self.language.title(),
            "E-c",
            f"2018-E-c-{langcodes.find(self.language.lower()).language.title()}-{split_mapping[split]}.txt",
        )
        df = pd.read_csv(filename, sep="\t")

        emotions = list(df.columns[2:])
        sorted_emotions = sorted(emotions)
        emotion_inds = [emotions.index(e) for e in sorted_emotions]
        texts = df.Tweet.values.tolist()
        ids = df.ID.values.tolist()

        labels = torch.tensor(df.iloc[:, 2:].values[:, emotion_inds]).float()

        return {
            _id: dict(text=self.preprocessor(t), original_text=t, label=l)
            for _id, t, l in zip(ids, texts, labels)
        }, sorted_emotions


class GoEmotions(TextDatasetWithPriors):
    """Plain text dataset for `GoEmotions`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDatasetWithPriors` for attributes.
    """

    multilabel = True
    annotator_labels = True
    name = "GoEmotions"
    source_domain = "Reddit"

    @property
    def label_similarity(self):
        if len(self.label_set) > 7:
            if not hasattr(self, "_label_similarity"):
                corrs = np.load(os.path.join(self.root_dir, "full_corrs.npy"))
                self._label_similarity = (
                    pd.DataFrame(
                        corrs, index=self.label_set, columns=self.label_set
                    )
                    / 2
                    + 0.5
                )
        else:
            self._label_similarity = PLUTCHIK_EMOTION_SIMILARITY

        return self._label_similarity

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = TextDatasetWithPriors.argparse_args() | dict(
            emotion_clustering_json=dict(
                type=str,
                help="JSON file with clustering of emotions",
            )
        )
        return args

    def __init__(self, emotion_clustering_json, *args, **kwargs):
        """Initializes dataset.

        Args:
            emotion_clustering_json: JSON file with clustering of emotions.
            Check `TextDatasetWithPriors` for other arguments.
        """
        self.emotion_clustering_json = emotion_clustering_json
        super().__init__(*args, **kwargs)

    def _multilabel_one_hot(
        self, labels: "np.ndarray", n_classes: int = 27
    ) -> torch.Tensor:
        """GoEmotions-specific label transformer to multilable one-hot,
        neutral emotion is discarded (represented as 0s)."""

        labels = [
            list(filter(lambda x: x < n_classes, map(int, lbl.split(","))))
            for lbl in labels
        ]
        new_labels = [
            torch.nn.functional.one_hot(
                torch.tensor(lbl, dtype=int), n_classes
            ).sum(0)
            for lbl in labels
        ]
        return torch.stack(new_labels)

    def _subset_emotions(
        self,
        annotations: dict[Any, dict[str, str | torch.Tensor]],
        emotions: list[str],
    ) -> list[str]:
        """Transforms emotions to a subset of emotions based on clustering
        in `emotion_clustering_json`. Each new label is union of old labels."""

        if not self.emotion_clustering_json:
            return emotions

        with open(self.emotion_clustering_json) as fp:
            clustering = json.load(fp)

        new_emotions = list(clustering)

        for annotation in annotations.values():
            for worker_id, label in annotation["label"].items():
                new_label = torch.zeros(len(new_emotions))

                for i, emotion in enumerate(new_emotions):
                    for old_emotion in clustering[emotion]:
                        new_label[i] += label[emotions.index(old_emotion)]

                annotation["label"][worker_id] = new_label.clamp(0, 1)

        return new_emotions

    def _load_data(
        self, split: str
    ) -> tuple[dict[Any, dict[str, str | torch.Tensor]], list[str]]:
        ## read emotions from file
        emotion_fn = os.path.join(self.root_dir, "emotions.txt")
        emotions = pd.read_csv(emotion_fn, header=None)[0].values.tolist()[
            :-1
        ]  # gets rid of neutral emotion

        ## read aggregated labels from file
        filename = os.path.join(self.root_dir, f"{split}.tsv")
        df = pd.read_csv(filename, sep="\t", header=None)

        ids = df.iloc[:, -1].values.tolist()
        aggr_labels = {
            _id: y
            for _id, y in zip(
                ids,
                self._multilabel_one_hot(
                    df.iloc[:, 1].values, len(emotions)
                ).float(),
            )
        }

        if self.annotation_mode == "aggregate":
            annotations = {
                _id: dict(
                    text=self.preprocessor(text),
                    original_text=text,
                    label={"aggregate": aggr_labels[_id]},
                )
                for _id, text in zip(ids, df.iloc[:, 0].values)
            }
            self.annotators = set()

        else:
            ## read annotator labels from file
            filenames = [
                os.path.join(self.root_dir, f"goemotions_{i}.csv")
                for i in range(1, 4)
            ]
            df = pd.concat([pd.read_csv(fn) for fn in filenames])
            df = df[df["id"].isin(set(ids))]
            df["labels"] = [
                [row[lbl] for lbl in emotions] for _, row in df.iterrows()
            ]

            groupby = df[["text", "rater_id", "id", "labels"]].groupby("id")
            annotations = groupby.agg(
                {
                    "text": lambda x: x.iloc[0],
                    "rater_id": lambda x: x.tolist(),
                    "labels": lambda x: x.tolist(),
                }
            )

            annotations = {
                _id: dict(
                    text=self.preprocessor(text),
                    original_text=text,
                    label={
                        worker_id: torch.tensor(labels).float()
                        for worker_id, labels in zip(rater_ids, label_list)
                    }
                    | {"aggregate": aggr_labels[_id]},
                )
                for _id, text, rater_ids, label_list in annotations.itertuples()
            }

            self.annotators = set(df["rater_id"].unique())

        emotions = self._subset_emotions(annotations, emotions)

        return annotations, emotions


class MFRC(TextDatasetWithPriors):
    """Plain text dataset for `MFRC`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDatasetWithPriors` for attributes.
    """

    multilabel = True
    annotator_labels = True
    name = "MFRC"
    source_domain = "Reddit"

    @property
    def label_similarity(self):
        if not hasattr(self, "_label_similarity"):
            self._label_similarity = pd.read_csv(
                os.path.join(self.root_dir, "mfq_corrs.csv"), index_col=0
            )
        return self._label_similarity

    def _load_data(self, split: str) -> tuple[
        dict[Any, dict[str, str | torch.Tensor | dict[str, torch.Tensor]]],
        list[str],
    ]:
        # { id1: { "text": "lorem ipsum", "label": {
        #   "ann1": torch.tensor([4]), "ann2": torch.tensor([3, 4]), "aggregate": torch.tensor([4]),
        # }, ... }, ... }

        # only train available in the dataset, contains entire dataset
        dataset = load_dataset("USC-MOLA-Lab/MFRC", split="train")

        with open(os.path.join(self.root_dir, "splits.yaml"), "r") as fp:
            text2id = yaml.safe_load(fp)[split]

        label_set = set()
        annotations = {}
        for e in dataset:
            id = text2id.get(e["text"], None)
            if id is None:
                # from another split
                continue

            labels = e["annotation"].split(",")
            if len(labels) > 1 and (
                "Non-Moral" in labels or "Thin Morality" in labels
            ):
                # https://arxiv.org/pdf/2208.05545v2 Appendix A.2.1:
                # Thin Morality only if no other label,
                # Non-Moral if no other label and not Thin Morality
                # so it cannot be that either is present and more than one modality
                continue
            elif labels[0] == "Non-Moral" or labels[0] == "Thin Morality":
                labels = []

            label_set.update(labels)

            if id not in annotations:
                annotations[id] = {
                    "text": self.preprocessor(e["text"]),
                    "original_text": e["text"],
                    "label": {e["annotator"]: labels},
                }
            else:
                annotations[id]["label"][e["annotator"]] = labels

        label_set = sorted(label_set)
        mlb = MultiLabelBinarizer().fit([label_set])
        for id in annotations:
            for annotator, label in annotations[id]["label"].items():
                annotations[id]["label"][annotator] = torch.tensor(
                    mlb.transform([label])[0]
                ).float()

            annotations[id]["label"]["aggregate"] = (
                (
                    sum(annotations[id]["label"].values())
                    / len(annotations[id]["label"])
                )
                >= 0.5
            ).float()

        return annotations, label_set


class MMLUPro(TextDatasetWithPriors):
    """Plain text dataset for `MMLU-Pro`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDatasetWithPriors` for attributes.
    """

    multilabel = False
    annotator_labels = False
    name = "MMLU-Pro"
    source_domain = "Many"
    label_similarity = None

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = TextDatasetWithPriors.argparse_args() | dict(
            delimiter=dict(
                type=str,
                default="\n",
                help="delimiter for options",
            )
        )
        del args["root_dir"]
        return args

    def __init__(self, delimiter: str = "\n", *args, **kwargs):
        self.delimiter = delimiter
        super().__init__(root_dir=None, *args, **kwargs)

    def _load_data(self, split: str) -> tuple[
        dict[Any, dict[str, str | torch.Tensor]],
        list[str],
    ]:
        # { id1: { "text": "lorem ipsum", "label": "label1", ... }, ... }
        # 12k in test, 70 in dev, no train

        if split in ("train", "dev"):
            split = "validation"

        dataset = load_dataset("TIGER-Lab/MMLU-Pro")[split]

        annotations = {}
        max_answers = 0
        for e in dataset:
            text = [e["question"]]
            text = [
                f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(e["options"])
            ]
            text = e["question"] + "\n" + self.delimiter.join(text)

            max_answers = max(max_answers, len(e["options"]))

            annotations[str(e["question_id"])] = {
                "text": self.preprocessor(text),
                "original_text": text,
                "label": torch.tensor(e["answer_index"]),
            }

        return annotations, [chr(ord('A') + i) for i in range(max_answers)]


class Boxes(TextDatasetWithPriors):
    """Plain text dataset for `Boxes`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance."""

    multilabel = True
    annotator_labels = False
    name = "Boxes Dataset"
    source_domain = "None"
    label_similarity = None

    @staticmethod
    def argparse_args():
        return TextDatasetWithPriors.argparse_args() | dict(
            subset=dict(
                type=str,
                choices=[
                    "few_shot_boxes_nso_exp2_max3",
                    "few_shot_boxes_nso_exp2_max3_move_contents",
                    # "few_shot_boxes_nso_exp2_max3_ambiref",
                ],
                default="few_shot_boxes_nso_exp2_max3_move_contents",
                help="subset to load",
            )
        )

    def __init__(self, subset, *args, **kwargs):
        self.subset = subset
        super().__init__(*args, **kwargs)

    def _load_data(self, split: str) -> tuple[
        dict[Any, dict[str, str | torch.Tensor]],
        list[str],
    ]:
        # { id1: { "text": "lorem ipsum", "label": "label1", ... }, ... }

        if split == "train":
            # use smaller subset for prompting
            split = "train-subset"
        elif split == "test":
            split = "test-subsample-states"

        df = pd.read_json(
            os.path.join(
                self.root_dir,
                self.subset,
                f"{split}-t5.jsonl",
            ),
            lines=True,
        )

        with open(os.path.join(self.root_dir, "objects.txt")) as fp:
            all_items = fp.readline().strip().split(",")

        annotations = {}
        same_id_cnt = 0
        prev_id = None
        for _, row in df.iterrows():
            special_token_idx = row["sentence_masked"].find("<extra_id_0>")
            sentence = row["sentence_masked"][:special_token_idx].strip()

            items = row["masked_content"][len("<extra_id_0>") :]
            items = [l.strip() for l in items.replace("the", "").split("and")]

            if row["sample_id"] == prev_id:
                same_id_cnt += 1
            else:
                same_id_cnt = 0
                prev_id = row["sample_id"]

            # a lot of examples share id
            annotations[f"{row['sample_id']}-{same_id_cnt}"] = {
                "text": self.preprocessor(sentence),
                "original_text": sentence,
                "label": items,
            }

        mlb = MultiLabelBinarizer().fit([all_items])
        for id in annotations:
            annotations[id]["label"] = torch.tensor(
                mlb.transform([annotations[id]["label"]])[0]
            ).float()

        return annotations, all_items


class MSPPodcast(TextDatasetWithPriors):
    """Plain text dataset for `MSP Podcast`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance."""

    @staticmethod
    def argparse_args():
        return TextDatasetWithPriors.argparse_args() | dict(
            multilabel=dict(
                action="store_true",
                help="whether to load as multilabel",
            ),
            annotator_labels=dict(
                action="store_true",
                help="whether to load annotator labels",
            ),
        )

    name = "MSP-Podcast-v1.11"
    source_domain = "Audio"
    # set here to avoid "can't instantiate abstract class" error
    annotator_labels = True
    multilabel = False
    # TODO: find similarity matrix
    label_similarity = None

    def __init__(
        self, multilabel=False, annotator_labels=False, *args, **kwargs
    ):
        self.multilabel = multilabel
        self.annotator_labels = annotator_labels
        super().__init__(*args, **kwargs)

    def _load_data(self, split):
        # { id1: { "text": "lorem ipsum", "label": {
        #   "ann1": torch.tensor([4]), "ann2": torch.tensor([3, 4]), "aggregate": torch.tensor([4]),
        # }, ... }, ... }

        transcript_fn = os.path.join(self.root_dir, "transcripts.csv")
        ann_label_fn = os.path.join(
            self.root_dir, "Labels", "labels_detailed.csv"
        )
        split_label_fn = os.path.join(
            self.root_dir, "Labels", "labels_consensus.csv"
        )

        ann_df = pd.read_csv(ann_label_fn, index_col=0)
        # remove .wav from ID
        ann_df.index = ann_df.index.str.replace(".wav", "")

        split_df = pd.read_csv(
            split_label_fn,
            index_col="FileName",
            usecols=["FileName", "Split_Set"],
        )
        # remove .wav from ID
        split_df.index = split_df.index.str.replace(".wav", "")
        split = {"train": "Train", "dev": "Development", "test": "Test1"}[split]
        split_ids = set(split_df.index[split_df["Split_Set"] == split])

        # drop ids that are not in split
        ann_df = ann_df[ann_df.index.isin(split_ids)]
        cols = [
            "worker_id",
            "single-label",
            "multiple-labels",
            "a",
            "v",
            "d",
            "null",
        ]
        # break EmoDetail into columns at ";"
        ann_df[cols] = ann_df.EmoDetail.str.split(";", expand=True)
        # keep only relevant columns
        ann_df = ann_df[cols[:3]]

        unique_ids = ann_df.index.unique()
        annotations = {}
        transcripts_df = pd.read_csv(transcript_fn, index_col="id")
        for _id in unique_ids:
            text = transcripts_df.loc[_id, "text"]
            if text and text is not np.nan:
                annotations[_id] = {
                    "text": self.preprocessor(text),
                    "original_text": text,
                    "label": {},
                }

        label_set = set()

        for _id, row in tqdm(
            ann_df.iterrows(), desc="Processing labels", total=len(ann_df)
        ):
            if _id not in annotations:
                continue

            if self.multilabel:
                other_idx = row["multiple-labels"].find("Other")
                if other_idx == -1:
                    label = row["multiple-labels"].split(",")
                else:
                    label = row["multiple-labels"][:other_idx].split(",")
                label = [l.strip() for l in label if l.strip()]
                label_set.update(label)
            else:
                label = row["single-label"].strip()
                if not label or "other" in label.lower():
                    continue
                label_set.add(label)

            annotations[_id]["label"][row["worker_id"]] = label

        ids_to_remove = []
        if self.multilabel:
            label_set = sorted(label_set.difference(["Neutral"]))
            mlb = MultiLabelBinarizer().fit([label_set])
            for _id in annotations:
                if "label" not in annotations[_id]:
                    ids_to_remove.append(_id)
                    continue
                for worker_id, label in annotations[_id]["label"].items():
                    annotations[_id]["label"][worker_id] = torch.tensor(
                        mlb.transform([label])[0]
                    ).float()

                annotations[_id]["label"]["aggregate"] = (
                    (
                        sum(annotations[_id]["label"].values())
                        / len(annotations[_id]["label"])
                    )
                    >= 0.5
                ).float()
        else:
            label_set = sorted(label_set)
            for _id in annotations:
                if not annotations[_id]["label"]:
                    ids_to_remove.append(_id)
                    continue

                labels = []
                for worker_id, label in annotations[_id]["label"].items():
                    label = label_set.index(label)
                    annotations[_id]["label"][worker_id] = torch.tensor(
                        label
                    ).float()
                    labels.append(label)

                # use most frequent label as aggregate
                cnt = Counter(labels)
                annotations[_id]["label"]["aggregate"] = torch.tensor(
                    int(cnt.most_common(1)[0][0])
                ).float()

        for _id in ids_to_remove:
            del annotations[_id]

        if not self.annotator_labels:
            for _id in annotations:
                annotations[_id]["label"] = annotations[_id]["label"][
                    "aggregate"
                ]

        return annotations, label_set


class QueerReclaimLex(TextDatasetWithPriors):
    """Plain text dataset for `QueerReclaimLex`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance."""

    # Instances are labeled for two definitions of harm depending on speaker identity. Scores are one of {0, 0.5, 1} where 0 means no harm, .5 means uncertain and 1 means harmful.
    # - `HARMFUL_IN`: Whether the post is harmful, given that the author is an *ingroup* member
    # - `HARMFUL_OUT`: Whether the post is harmful, given that the author is an *outgroup* member.

    # Each type of harm has variables for 4 different values. The same can be extended for `HARMFUL_OUT`.
    # - `HARMFUL_IN_1` denotes annotator 1's score, `HARMFUL_IN_2` for annotator 2's score
    # - `HARMFUL_IN_mu` for the mean of the two annotator's harm scores
    # - `HARMFUL_IN_gold` is a binary variable reflecting whether the harm score's mean is above a threshold of 0.5.

    annotator_labels = True
    name = "QueerReclaimLex"
    source_domain = "Twitter"
    label_similarity = None

    @staticmethod
    def argparse_args():
        return TextDatasetWithPriors.argparse_args() | dict(
            type=dict(
                type=str,
                choices=["in", "out", "both"],
                default="both",
                help="type of harm to load",
                searchable=True,
            ),
            discard_ambiguous=dict(
                action="store_true",
                help="discard ambiguous examples (label == 0.5)",
            ),
        )

    @property
    def multilabel(self):
        return self.type == "both"

    def __init__(
        self,
        type: str = "both",
        discard_ambiguous: bool = False,
        *args,
        **kwargs,
    ):
        assert type in (
            "in",
            "out",
            "both",
        ), "type must be 'in', 'out', or 'both'"
        self.type = type
        self.discard_ambiguous = discard_ambiguous
        super().__init__(*args, **kwargs)

    def _load_data(self, split: str) -> tuple[
        dict[Any, dict[str, str | torch.Tensor]],
        list[str],
    ]:
        # read split
        with open(os.path.join(self.root_dir, "balanced-splits.json")) as fp:
            split_ids = json.load(fp)[split]

        df = pd.read_csv(
            os.path.join(self.root_dir, "QueerReclaimLex.csv"), index_col=0
        )
        # id is combination of col "template_idx" and col "term", e.g. 1-queer.
        # to access, we will create a dict from col "template" to this id
        df["id"] = df["template_idx"].astype(str) + "-" + df["term"]
        # filter by split
        df = df[df["id"].astype(str).isin(split_ids)]
        ids = {row["template"]: row["id"] for _, row in df.iterrows()}

        # read all Annotator#.xlsx from "Annotations" folder
        files = os.listdir(os.path.join(self.root_dir, "Annotations"))
        # use regex to match Annotator#.xlsx
        files = sorted([f for f in files if re.match(r"Annotator\d+.xlsx", f)])
        # read all files
        dfs = [
            pd.read_excel(
                os.path.join(self.root_dir, "Annotations", f),
                sheet_name="slurs",
            )
            for f in files
        ] + [
            pd.read_excel(
                os.path.join(self.root_dir, "Annotations", f),
                sheet_name="identity terms",
            )
            for f in files
        ]
        # remove columns where HARMFUL_IF_IN, HARMFUL_IF_OUT is not 0, 0.5, 1
        for i, df in enumerate(dfs):
            df = df[
                df["HARMFUL_IF_IN"].isin([0, 0.5, 1])
                & df["HARMFUL_IF_OUT"].isin([0, 0.5, 1])
            ]
            dfs[i] = df

        # add annotator id to each dataframe
        for i, df in enumerate(dfs):
            df["annotator_id"] = os.path.splitext(files[i % len(files)])[0]

        # concatenate all dataframes
        df = pd.concat(dfs)

        annotations = {}
        for i, row in df.iterrows():
            if self.type == "in":
                label = torch.tensor(row["HARMFUL_IF_IN"]).float()
            elif self.type == "out":
                label = torch.tensor(row["HARMFUL_IF_OUT"]).float()
            else:
                label = torch.tensor(
                    [row["HARMFUL_IF_IN"], row["HARMFUL_IF_OUT"]]
                ).float()

            _id = ids.get(row["POST"], None)
            if _id is None:
                continue

            if _id not in annotations:
                annotations[_id] = {
                    "text": self.preprocessor(row["POST"]),
                    "original_text": row["POST"],
                    "label": {row["annotator_id"]: label},
                }
            else:
                annotations[_id]["label"][row["annotator_id"]] = label

        for _id in list(annotations):
            aggregate = sum(annotations[_id]["label"].values()) / len(
                annotations[_id]["label"]
            )
            # round aggregate to 0 or 1
            if self.discard_ambiguous and aggregate == 0.5:
                annotations.pop(_id)
            else:
                annotations[_id]["label"]["aggregate"] = (
                    aggregate > 0.5
                ).float()

        label_set = (
            ["ingroup harm", "outgroup harm"]
            if self.multilabel
            else ["no harm", "harm"]
        )

        return annotations, label_set


class Hatexplain(TextDatasetWithPriors):
    """Plain text dataset for `Hatexplain`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance."""

    multilabel = False
    annotator_labels = True
    name = "Hatexplain"
    source_domain = "None"
    label_similarity = None

    @staticmethod
    def argparse_args():
        args = TextDatasetWithPriors.argparse_args()
        del args["root_dir"]
        return args

    def __init__(self, *args, **kwargs):
        super().__init__(root_dir=None, *args, **kwargs)

    def _load_data(self, split):
        split = {
            "train": "train",
            "dev": "validation",
            "test": "test",
        }[split]
        dataset = load_dataset("Hate-speech-CNERG/hatexplain", split=split)

        annotations = {}

        for e in dataset:
            text = " ".join(e["post_tokens"])
            annotations[e["id"]] = {
                "text": self.preprocessor(text),
                "original_text": text,
                "label": {},
            }
            for ann_id, label in zip(
                e["annotators"]["annotator_id"], e["annotators"]["label"]
            ):
                annotations[e["id"]]["label"][ann_id] = torch.tensor(
                    label
                ).float()

        # use most frequent label as aggregate
        for _id in annotations:
            cnt = Counter(
                [x.item() for x in annotations[_id]["label"].values()]
            )
            annotations[_id]["label"]["aggregate"] = torch.tensor(
                int(cnt.most_common(1)[0][0])
            ).float()

        return annotations, ["hate", "normal", "offensive"]


class TREC(TextDatasetWithPriors):
    multilabel = False
    annotator_labels = False
    name = "Text REtrieval Conference"
    source_domain = "None"
    label_similarity = None

    @staticmethod
    def argparse_args():
        args = TextDatasetWithPriors.argparse_args()
        del args["root_dir"]
        return args

    def __init__(self, *args, **kwargs):
        super().__init__(root_dir=None, *args, **kwargs)

    def _load_data(self, split):
        # 'ABBR' (0): Abbreviation.
        # 'ENTY' (1): Entity.
        # 'DESC' (2): Description and abstract concept.
        # 'HUM' (3): Human being.
        # 'LOC' (4): Location.
        # 'NUM' (5): Numeric value.

        split = {
            "train": "train",
            "dev": "test",
            "test": "test",
        }[split]

        dataset = load_dataset("trec", split=split)
        annotations = {}
        for i, e in enumerate(dataset):
            text = e["text"]
            annotations[str(i)] = {
                "text": self.preprocessor(text),
                "original_text": text,
                "label": torch.tensor(e["coarse_label"]).float(),
            }

        return annotations, [
            "Abbreviation",
            "Entity",
            "Description",
            "Human",
            "Location",
            "Numeric",
        ]


class MovieLens(TextDatasetWithPriors):
    """Plain text dataset for `MovieLens`. Input is movie title and summary,
    labels are movie genres (multi-label). Class doesn't use from_namespace
    decorator, so it can be used for inheritance."""

    multilabel = True
    annotator_labels = False
    name = "MovieLens"
    source_domain = "None"
    label_similarity = None

    def _load_data(self, split) -> tuple[list[dict[str, Any]], list[str]]:
        """Load MovieLens movies joined with IMDb details.

        - Left-joins movies on imdb_details by movieId.
        - If imdb_details is missing for some titles, attempts to backfill imdbId from links.csv.
        """

        def _load_csv_dict(path: str) -> list[dict[str, str]]:
            with open(path, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))

        def _split_pipe_list(s: str) -> list[str]:
            if s is None:
                return []
            s = s.strip()
            if not s:
                return []
            return [part.strip() for part in s.split("|") if part.strip()]

        movies_path = os.path.join(self.root_dir, "movies.csv")
        imdb_details_path = os.path.join(self.root_dir, "imdb_details.csv")
        links_path = os.path.join(self.root_dir, "links.csv")

        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Missing movies.csv at {movies_path}")

        movies_rows = _load_csv_dict(movies_path)

        imdb_by_movie = {}
        if imdb_details_path and os.path.exists(imdb_details_path):
            for row in _load_csv_dict(imdb_details_path):
                imdb_by_movie[row["movieId"]] = row

        links_by_movie = {}
        if links_path and os.path.exists(links_path):
            for row in _load_csv_dict(links_path):
                links_by_movie[row["movieId"]] = row

        label_set = set([])
        for m in movies_rows:
            genres = _split_pipe_list(m.get("genres", ""))
            label_set.update(genres)
        label_set.discard("(no genres listed)")
        label_set = sorted(label_set)

        annotations = {}
        for m in movies_rows:
            movie_id = m.get("movieId", None)
            if movie_id is None:
                continue
            title = m.get("title", "")
            if not title:
                continue
            genres = _split_pipe_list(m.get("genres", ""))
            imdb_row = imdb_by_movie.get(movie_id, {})
            plot_summaries = imdb_row.get("plot_summaries", "")
            if not plot_summaries:
                continue
            plot_list = _split_pipe_list(plot_summaries)
            combined_summary = plot_list[0] if plot_list else ""

            annotations[f"{title} - {movie_id}"] = {
                "text": self.preprocessor(combined_summary),
                "original_text": combined_summary,
                "label": torch.tensor(
                    [1 if genre in genres else 0 for genre in label_set]
                ).float(),
            }

        ## splits
        if split == "train":
            annotations = {
                k: v
                for i, (k, v) in enumerate(annotations.items())
                if i % 7 > 1
            }
        elif split == "dev":
            annotations = {
                k: v
                for i, (k, v) in enumerate(annotations.items())
                if i % 7 == 0
            }
        elif split == "test":
            annotations = {
                k: v
                for i, (k, v) in enumerate(annotations.items())
                if i % 7 == 1
            }
        else:
            raise ValueError(
                f"Invalid split {split}, must be one of train, dev, test"
            )

        return annotations, label_set


class PersuasionForGood(TextDatasetWithPriors):
    """Plain text dataset for `PersuasionForGood`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance."""

    regression = True
    multilabel = False
    annotator_labels = False
    name = "PersuasionForGood"
    source_domain = "None"
    label_similarity = None

    @staticmethod
    def argparse_args():
        return TextDatasetWithPriors.argparse_args() | dict(
            label_type=dict(
                type=str,
                choices=[
                    "persuader_donation",
                    "persuadee_donation",
                    "persuader_success",
                    "donation",
                ],
                default="persuader_success",
                help="type of label to load",
            )
        )

    def __init__(self, label_type: str, *args, **kwargs):
        self.label_type = label_type
        assert self.label_type in (
            "persuader_donation",
            "persuadee_donation",
            "persuader_success",
            "donation",
        ), f"label_type must be one of 'persuader_donation', 'persuadee_donation', 'persuader_success', or 'donation', got {self.label_type}"
        if self.label_type == "persuader_success":
            self.regression = False
        super().__init__(*args, **kwargs)

    def _build_label(self, persuader_donation, persuadee_donation):
        if self.label_type == "persuader_donation":
            return torch.tensor(persuader_donation).float()
        elif self.label_type == "persuadee_donation":
            return torch.tensor(persuadee_donation).float()
        elif self.label_type == "persuader_success":
            return torch.tensor(int(persuadee_donation > 0)).float()
        elif self.label_type == "donation":
            return torch.tensor(persuader_donation + persuadee_donation).float()

    def _build_label_set(self):
        if self.label_type == "persuader_success":
            return ["unsuccessful", "successful"]
        else:
            return "donation amount"

    def _load_data(self, split):
        dialogs_csv = (
            Path(self.root_dir) / "data" / "FullData" / "full_dialog.csv"
        )
        info_csv = Path(self.root_dir) / "data" / "FullData" / "full_info.csv"

        dialogs = {}
        with open(dialogs_csv) as fp:
            reader = csv.reader(fp)
            headers = next(reader)
            headers[0] = "turn_idx"
            for row in reader:
                row_dict = {k: v for k, v in zip(headers, row)}
                dialog_id = row_dict["B2"]
                turn_idx = int(row_dict["turn_idx"])
                dialogs.setdefault(dialog_id, {})[turn_idx] = row_dict["Unit"]

        info = {}
        with open(info_csv) as fp:
            reader = csv.reader(fp)
            headers = next(reader)
            for row in reader:
                row_dict = {k: v for k, v in zip(headers, row)}
                dialog_id = row_dict["B2"]
                info.setdefault(dialog_id, {})[
                    (
                        "persuader_donation"
                        if row_dict["B4"] == "0"
                        else "persuadee_donation"
                    )
                ] = float(row_dict["B6"])

        annotations = {}
        for dialog_id, turns in dialogs.items():
            for turn_idx in sorted(turns):
                dialog_turn_idx = dialog_id + "--" + str(turn_idx)
                text_up_to_turn = "\n".join(
                    [f"Turn {i}: " + turns[i] for i in range(int(turn_idx) + 1)]
                ).strip()

                annotations[dialog_turn_idx] = {
                    "text": self.preprocessor(text_up_to_turn),
                    "original_text": text_up_to_turn,
                    "label": self._build_label(
                        info[dialog_id].get("persuader_donation", 0),
                        info[dialog_id].get("persuadee_donation", 0),
                    ),
                }

        # splits are by dialog_id prefix
        train_dialogs = {k for i, k in enumerate(dialogs.keys()) if i % 5 > 1}
        dev_dialogs = {k for i, k in enumerate(dialogs.keys()) if i % 5 == 0}
        test_dialogs = {k for i, k in enumerate(dialogs.keys()) if i % 5 == 1}

        if split == "train":
            annotations = {
                k: v
                for k, v in annotations.items()
                if k.split("--")[0] in train_dialogs
            }
        elif split == "dev":
            annotations = {
                k: v
                for k, v in annotations.items()
                if k.split("--")[0] in dev_dialogs
            }
        elif split == "test":
            annotations = {
                k: v
                for k, v in annotations.items()
                if k.split("--")[0] in test_dialogs
            }

        return annotations, self._build_label_set()


class LAPDRationales(TextDatasetWithPriors):
    """Plain text dataset for `LAPD Rationales`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance."""

    multilabel = False
    annotator_labels = False
    name = "LAPD-Rationales"
    source_domain = "None"
    label_similarity = pd.DataFrame(
        [
            [1.0, 0.50, 0.33, 0.25, 0.20],
            [0.50, 1.0, 0.50, 0.33, 0.25],
            [0.33, 0.50, 1.0, 0.50, 0.33],
            [0.25, 0.33, 0.50, 1.0, 0.50],
            [0.20, 0.25, 0.33, 0.50, 1.0],
        ],
        columns=[
            "very disrespectful",
            "disrespectful",
            "neutral",
            "respectful",
            "very respectful",
        ],
        index=[
            "very disrespectful",
            "disrespectful",
            "neutral",
            "respectful",
            "very respectful",
        ],
    )
    regression = True

    def regression_value_for_label(self, label: str) -> float:
        mapping = {l: i for i, l in enumerate(self.label_set, start=1)}
        return mapping[label]

    @staticmethod
    def argparse_args():
        return TextDatasetWithPriors.argparse_args() | dict(
            role=dict(
                type=str,
                choices=["officer", "civilian"],
                default="officer",
                help="Perspective to load",
                searchable=True,
                metadata=dict(name=True),
            ),
            remove_confounders=dict(
                type=bool,
                default=False,
                help="Whether to remove confounding factors from text (aka respectful, very disrespectful, 4, 2, etc.)",
            ),
        )

    def __init__(self, role, remove_confounders=False, *args, **kwargs):
        self.role = role
        self.remove_confounders = remove_confounders
        super().__init__(*args, **kwargs)

    def _load_data(self, split: str) -> tuple[
        dict[Any, dict[str, str | torch.Tensor | dict[str, torch.Tensor]]],
        list[str],
    ]:
        def iter_csv(filepath: str | Path) -> Iterator[dict[str, str]]:
            """Yield rows from a CSV file as dictionaries.

            - Uses the first line as headers.
            - Handles quoted fields and commas inside quotes.
            - Streams rows to avoid loading the entire file in memory.
            """
            path = Path(filepath)
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Ensure a standard dict (not OrderedDict) and type as str->str
                    yield dict(row)  # type: ignore[arg-type]

        annotations = {}
        for row in iter_csv(
            Path(self.root_dir) / "phase2b_data_sep9_processed.csv"
        ):

            text = (
                row["person:Police:Primary_respect_rationale"]
                if self.role == "officer"
                else row["person:Civilian:Driver_respect_rationale"]
            )
            label = (
                row["person:Police:Primary_respect_rating"]
                if self.role == "officer"
                else row["person:Civilian:Driver_respect_rating"]
            )

            if not text or text.strip() == "" or not label.strip():
                continue

            if self.remove_confounders:
                # remove words respectful, disrespectful, very, etc.
                text = re.sub(
                    r"\b(very\s+)?(dis)?respectful\b",
                    "[MUTE]",
                    text,
                    flags=re.I,
                )
                # remove numbers 1-5
                text = re.sub(r"\b[1-5]\b", "[MUTE]", text)
                # remove extra spaces
                text = re.sub(r"\s+", " ", text).strip()

            annotations[row["job_id"]] = {
                "text": self.preprocessor(text),
                "original_text": text,
                "label": torch.tensor(float(label)).float() - 1,
            }

        train_idx = 25
        dev_idx = 5 + train_idx
        if split == "train":
            annotations = {
                k: v
                for i, (k, v) in enumerate(annotations.items())
                if i < train_idx
            }
        elif split == "dev":
            annotations = {
                k: v
                for i, (k, v) in enumerate(annotations.items())
                if i >= train_idx and i < dev_idx
            }
        else:
            annotations = {
                k: v
                for i, (k, v) in enumerate(annotations.items())
                if i >= dev_idx
            }

        return annotations, [
            "Very Disrespectful",
            "Disrespectful",
            "Neutral",
            "Respectful",
            "Very Respectful",
        ]
