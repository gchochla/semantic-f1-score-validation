from typing import Any

import torch
from legm import from_namespace

from sef1_validation.base_datasets import TokenizationMixin
from sef1_validation.benchmarks import (
    SemEval2018Task1Ec,
    GoEmotions,
    MFRC,
    MMLUPro,
    Boxes,
    MSPPodcast,
    QueerReclaimLex,
    Hatexplain,
    TREC,
    MovieLens,
    PersuasionForGood,
    LAPDRationales,
)


class SemEval2018Task1EcDataset(SemEval2018Task1Ec):
    """Plain text dataset for `SemEval 2018 Task 1: Affect in Tweets`
    (https://competitions.codalab.org/competitions/17751). Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `SemEval2018Task1Ec`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SemEval2018Task1EcDatasetForTransformers(
    TokenizationMixin, SemEval2018Task1Ec
):
    """Dataset with encodings for `transformers`
    for `SemEval 2018 Task 1: Affect in Tweets`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            SemEval2018Task1Ec.argparse_args()
            | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch


class GoEmotionsDataset(GoEmotions):
    """Plain text dataset for `GoEmotions`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `GoEmotions`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GoEmotionsDatasetForTransformers(TokenizationMixin, GoEmotions):
    """Dataset with encodings for `transformers`
    for `GoEmotions`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return GoEmotions.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch


class MFRCDataset(MFRC):
    """Plain text dataset for `MFRC`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `MFRC`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MFRCDatasetForTransformers(TokenizationMixin, MFRC):
    """Dataset with encodings for `transformers`
    for `MFRC`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return MFRC.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch


class MMLUProDataset(MMLUPro):
    """Plain text dataset for `MMLUPro`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `MMLUPro`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MMLUProDatasetForTransformers(TokenizationMixin, MMLUPro):
    """Dataset with encodings for `transformers`
    for `MMLUPro`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return MMLUPro.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch


class BoxesDataset(Boxes):
    """Plain text dataset for `Boxes`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `Boxes`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BoxesDatasetForTransformers(TokenizationMixin, Boxes):
    """Dataset with encodings for `transformers` for `Boxes`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return Boxes.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch


class MSPPodcastDataset(MSPPodcast):
    """Plain text dataset for `MSPPodcast`. Class uses"
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `MSPPodcast`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MSPPodcastDatasetForTransformers(TokenizationMixin, MSPPodcast):
    """Dataset with encodings for `transformers` for `MSPPodcast`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return MSPPodcast.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch


class QueerReclaimLexDataset(QueerReclaimLex):
    """Plain text dataset for `QueerReclaimLex`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `QueerReclaimLex`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QueerReclaimLexDatasetForTransformers(TokenizationMixin, QueerReclaimLex):
    """Dataset with encodings for `transformers` for `QueerReclaimLex`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            QueerReclaimLex.argparse_args() | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch


class HatexplainDataset(Hatexplain):
    """Plain text dataset for `Hatexplain`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `Hatexplain`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HatexplainDatasetForTransformers(TokenizationMixin, Hatexplain):
    """Dataset with encodings for `transformers` for `Hatexplain`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return Hatexplain.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch


class TRECDataset(TREC):
    """Plain text dataset for `TREC`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `TREC`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TRECDatasetForTransformers(TokenizationMixin, TREC):
    """Dataset with encodings for `transformers` for `TREC`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return TREC.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch


class MovieLensDataset(MovieLens):
    """Plain text dataset for `MovieLens`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `MovieLens`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MovieLensDatasetForTransformers(TokenizationMixin, MovieLens):
    """Dataset with encodings for `transformers` for `MovieLens`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return MovieLens.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch


class PersuasionForGoodDataset(PersuasionForGood):
    """Plain text dataset for `PersuasionForGood`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `PersuasionForGood`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PersuasionForGoodDatasetForTransformers(
    TokenizationMixin, PersuasionForGood
):
    """Dataset with encodings for `transformers` for `PersuasionForGood`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            PersuasionForGood.argparse_args()
            | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch


class LAPDRationalesDataset(LAPDRationales):
    """Plain text dataset for `LAPD Rationales`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `LAPD Rationales`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LAPDRationalesDatasetForTransformers(TokenizationMixin, LAPDRationales):
    """Dataset with encodings for `transformers` for `LAPD Rationales`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            LAPDRationales.argparse_args() | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        """Collate function for `transformers`."""
        batch = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
        batch["encoding"] = self.batch_tokenize(batch["text"])
        batch["label"] = torch.stack(batch["label"])
        return batch
