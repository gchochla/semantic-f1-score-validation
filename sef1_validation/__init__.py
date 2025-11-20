"""Top-level package for llm_subj.

This module avoids importing heavy optional dependencies (torch, transformers,
OpenAI, etc.) at import time to keep lightweight users (e.g., metrics tests)
snappy. Public attributes from submodules are provided via lazy imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # Convenience exports (lazily resolved)
    "LMForClassification",
    "OpenAIClassifier",
    "vLMForGeneration",
    "vLMForClassification",
    "PromptDataset",
    "PromptTextDataset",
    "OpenAIPromptTextDataset",
    "ReasonablenessPromptDataset",
    "OpenAIReasonablenessPromptTextDataset",
    "PromptEvaluator",
    "vPromptEvaluator",
    "APIPromptEvaluator",
    "ReasonablenessEvaluator",
    "vReasonablenessEvaluator",
    "APIReasonablenessEvaluator",
    "twitter_preprocessor",
    "reddit_preprocessor",
    "text_preprocessor",
    "CONSTANT_ARGS",
    "DATASETS",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name in {
        "LMForClassification",
        "OpenAIClassifier",
        "vLMForClassification",
        "vLMForGeneration",
    }:
        return getattr(import_module("llm_subj.models"), name)
    if name in {
        "PromptDataset",
        "PromptTextDataset",
        "OpenAIPromptTextDataset",
        "ReasonablenessPromptDataset",
        "OpenAIReasonablenessPromptTextDataset",
    }:
        return getattr(import_module("llm_subj.prompt_dataset"), name)
    if name in {
        "PromptEvaluator",
        "vPromptEvaluator",
        "APIPromptEvaluator",
        "ReasonablenessEvaluator",
        "vReasonablenessEvaluator",
        "APIReasonablenessEvaluator",
    }:
        return getattr(import_module("llm_subj.trainers"), name)
    if name in {"twitter_preprocessor", "reddit_preprocessor"}:
        return getattr(import_module("llm_subj.utils"), name)
    if name == "text_preprocessor":
        utils = import_module("llm_subj.utils")
        return dict(
            Twitter=getattr(utils, "twitter_preprocessor"),
            Reddit=getattr(utils, "reddit_preprocessor"),
            Plain=lambda *a, **k: (lambda x: x),
        )
    if name == "CONSTANT_ARGS":
        # Keep lightweight constants available without importing heavy modules
        return dict(
            seed=dict(
                type=int,
                help="random seed",
                metadata=dict(disable_comparison=True),
                searchable=True,
            ),
        )
    if name == "DATASETS":
        # Build lazily to avoid importing transformers, etc., on import
        dsm = import_module("llm_subj.datasets")
        return dict(
            SemEval=getattr(dsm, "SemEval2018Task1EcDataset"),
            GoEmotions=getattr(dsm, "GoEmotionsDataset"),
            MFRC=getattr(dsm, "MFRCDataset"),
            MMLUPro=getattr(dsm, "MMLUProDataset"),
            Boxes=getattr(dsm, "BoxesDataset"),
            MSPPodcast=getattr(dsm, "MSPPodcastDataset"),
            QueerReclaimLex=getattr(dsm, "QueerReclaimLexDataset"),
            Hatexplain=getattr(dsm, "HatexplainDataset"),
            TREC=getattr(dsm, "TRECDataset"),
            MovieLens=getattr(dsm, "MovieLensDataset"),
            PersuasionForGood=getattr(dsm, "PersuasionForGoodDataset"),
            LAPDRationales=getattr(dsm, "LAPDRationalesDataset"),
        )
    if name == "TOKENIZED_DATASETS":
        dsm = import_module("llm_subj.datasets")
        return dict(
            SemEval=getattr(dsm, "SemEval2018Task1EcDatasetForTransformers"),
            GoEmotions=getattr(dsm, "GoEmotionsDatasetForTransformers"),
            MFRC=getattr(dsm, "MFRCDatasetForTransformers"),
        )
    raise AttributeError(name)
