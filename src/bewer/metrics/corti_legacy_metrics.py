import string
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING

from error_align import error_align
from error_align.utils import OpType
from fuzzywuzzy import fuzz, process
from rapidfuzz.distance import Levenshtein

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric
from bewer.metrics.metrics import CER, WER
from bewer.preprocessing.context import set_pipeline

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.example import Example

METRIC_REGISTRY.register_metric(
    WER,
    "legacy_wer",
    tokenizer="legacy",
    normalizer="legacy",
)
METRIC_REGISTRY.register_metric(
    WER,
    "legacy_wer_uncased",
    tokenizer="legacy",
    normalizer="legacy_uncased",
)
METRIC_REGISTRY.register_metric(
    WER,
    "legacy_wer_uncased_no_punct",
    tokenizer="legacy",
    normalizer="legacy_uncased_no_punct",
)

METRIC_REGISTRY.register_metric(
    CER,
    "legacy_cer",
    tokenizer="legacy",
    normalizer="legacy",
)
METRIC_REGISTRY.register_metric(
    CER,
    "legacy_cer_uncased",
    tokenizer="legacy",
    normalizer="legacy_uncased",
)
METRIC_REGISTRY.register_metric(
    CER,
    "legacy_cer_uncased_no_punct",
    tokenizer="legacy",
    normalizer="legacy_uncased_no_punct",
)


class _KeywordAggregator(ExampleMetric):
    @cached_property
    def cer_keyword(self) -> float:
        """Get the character error rate for medical terms."""
        return self._keyword_metrics["cer_keyword"]

    @cached_property
    def total_distance(self) -> float:
        """Get the total Levenshtein distance for medical terms."""
        return self._keyword_metrics["total_distance"]

    @cached_property
    def total_length(self) -> float:
        """Get the total length of medical terms."""
        return self._keyword_metrics["total_length"]

    @cached_property
    def match_count(self) -> int:
        """Get the number of exactly matched medical terms."""
        return self._keyword_metrics["match_count"]

    @cached_property
    def relaxed_match_count(self) -> int:
        """Get the number of medical terms matched with relaxed criteria."""
        return self._keyword_metrics["relaxed_match_count"]

    @cached_property
    def total_terms(self) -> int:
        """Get the total number of medical terms."""
        return self._keyword_metrics["total_terms"]

    @cached_property
    def correct_terms(self) -> list[str]:
        """Get the list of correctly matched medical terms."""
        return self._keyword_metrics["correct_terms"]

    @cached_property
    def _keyword_metrics(self) -> dict[str, float]:
        """Calculate keyword-focused metrics for medical terms."""

        with set_pipeline(*self.src_metric.pipeline):
            total_distance = 0
            total_length = 0
            match_count = 0
            relaxed_match_count = 0
            correct_terms = []

            if "medical_terms" not in self.example.keywords:
                return dict(
                    cer_keyword=0.0,
                    total_distance=0.0,
                    total_length=0.0,
                    match_count=0,
                    relaxed_match_count=0,
                    total_terms=0,
                    correct_terms=[],
                )

            medical_terms = self.example.keywords["medical_terms"]
            max_n = max((len(term.tokens) for term in medical_terms), default=0)
            words = self.example.hyp.tokens.normalized
            ngram_matrix = [self._get_ngrams(words, n) for n in range(1, max_n + 1)]

            for term in medical_terms:
                term = term.joined(normalized=True)
                distance = self._term_distance(term, ngram_matrix=ngram_matrix)
                cer_score = distance / max(len(term), 1)  # Avoid division by zero
                if distance == 0:
                    match_count += 1
                    correct_terms.append(term)
                if cer_score <= self.src_metric.cer_threshold:
                    relaxed_match_count += 1
                total_distance += distance
                total_length += max(len(term), 1)

            cer_keyword = total_distance / total_length if total_length > 0 else 0

        return dict(
            cer_keyword=cer_keyword,
            total_distance=total_distance,
            total_length=total_length,
            match_count=match_count,
            relaxed_match_count=relaxed_match_count,
            total_terms=len(medical_terms),
            correct_terms=correct_terms,
        )

    @staticmethod
    def _get_ngrams(tokens: list[str], n: int) -> list[str]:
        """Get n-grams from a list of tokens."""
        if n < 1:
            raise ValueError("n must be a positive integer")
        if n == 1:
            return tokens
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def _term_distance(term: str, ngram_matrix: list[list[str]], retain_punc_list: list[str] = ["-", "/"]) -> int:
        """Calculate the Levenshtein distance between a term and a best match ngram."""
        # Select n-grams for the text
        n = len(term.split())
        ngrams = list(chain.from_iterable(ngram_matrix[:n]))

        # Find the best match using fuzzy matching
        best_match, _ = process.extractOne(term, ngrams, scorer=fuzz.ratio) if ngrams else ("", 0)

        # Define punctuation for the check as all punctuation except the ones we want to retain
        punct_to_remove = string.punctuation
        for p in retain_punc_list:
            punct_to_remove = punct_to_remove.replace(p, "")

        translator = str.maketrans("", "", punct_to_remove)
        punctuation_for_check = set(punct_to_remove)

        # Remove any punctuation (excluding retained punctuation)
        if not any(char in punctuation_for_check for char in term):
            best_match = best_match.translate(translator)

        best_match = best_match.lower()
        term = term.lower()

        # Calculate the Levenshtein distance
        return Levenshtein.distance(term, best_match)


@METRIC_REGISTRY.register("_legacy_kwa")
class KeywordAggregator(Metric):
    short_name = "kwa"
    long_name = "Keyword Aggregator"
    description = "Aggregates keyword-focused metrics for medical terms."
    example_cls = _KeywordAggregator

    def __init__(
        self,
        src: "Dataset",
        name: str = "_legacy_kwa",
        cer_threshold: float = 0.2,
        standardizer: str = "default",
        tokenizer: str = "legacy",
        normalizer: str = "legacy_uncased",
    ):
        """Initialize the CER Metric object."""
        self.pipeline = (standardizer, tokenizer, normalizer)
        self.cer_threshold = cer_threshold
        super().__init__(src, name)

    @cached_property
    def match_count(self) -> int:
        """Get the total number of exactly matched medical terms."""
        return sum([example.metrics.get(self.name).match_count for example in self._src_dataset])

    @cached_property
    def relaxed_match_count(self) -> int:
        """Get the total number of medical terms matched with relaxed criteria."""
        return sum([example.metrics.get(self.name).relaxed_match_count for example in self._src_dataset])

    @cached_property
    def total_terms(self) -> int:
        """Get the total number of medical terms."""
        return sum([example.metrics.get(self.name).total_terms for example in self._src_dataset])

    @cached_property
    def total_length(self) -> float:
        """Get the total length of medical terms."""
        return sum([example.metrics.get(self.name).total_length for example in self._src_dataset])

    @cached_property
    def total_distance(self) -> float:
        """Get the total Levenshtein distance of medical terms."""
        return sum([example.metrics.get(self.name).total_distance for example in self._src_dataset])

    @cached_property
    def correct_terms(self) -> list[str]:
        """Get the list of correctly matched medical terms."""
        return list(
            chain.from_iterable([example.metrics.get(self.name).correct_terms for example in self._src_dataset])
        )


@METRIC_REGISTRY.register("legacy_medical_word_accuracy")
class MTR(Metric):
    short_name = "MTR"
    long_name = "Medical Term Recall"
    description = (
        "Medical Term Recall (MTR) is computed as the number of correctly recognized medical terms "
        "divided by the total number of medical terms in the reference transcripts."
    )

    @cached_property
    def value(self) -> float:
        """Get the medical term recall."""
        if self._src_dataset.metrics._legacy_kwa.total_terms == 0:
            return 1.0
        return self._src_dataset.metrics._legacy_kwa.match_count / self._src_dataset.metrics._legacy_kwa.total_terms


@METRIC_REGISTRY.register("legacy_relaxed_medical_word_accuracy")
class RMTR(Metric):
    short_name = "Relaxed MTR"
    long_name = "Relaxed Medical Term Recall"
    description = (
        "Relaxed Medical Term Recall (RMTR) is computed as the number of terms recognized with a relaxed criteria "
        "(i.e., within a certain character error rate threshold) divided by the total number of medical terms in the "
        "reference transcripts."
    )

    @cached_property
    def value(self) -> float:
        """Get the medical term recall."""
        if self._src_dataset.metrics._legacy_kwa.total_terms == 0:
            return 1.0
        return (
            self._src_dataset.metrics._legacy_kwa.relaxed_match_count
            / self._src_dataset.metrics._legacy_kwa.total_terms
        )


@METRIC_REGISTRY.register("legacy_keyword_cer")
class KeywordCER(Metric):
    short_name = "Keyword CER"
    long_name = "Keyword Character Error Rate"
    description = (
        "Keyword Character Error Rate (Keyword CER) is computed as the character error rate for medical terms "
        "between the reference and hypothesis terms."
    )

    @cached_property
    def value(self) -> float:
        """Get the medical character error rate."""
        if self._src_dataset.metrics._legacy_kwa.total_length == 0:
            return 1.0
        return self._src_dataset.metrics._legacy_kwa.total_distance / self._src_dataset.metrics._legacy_kwa.total_length


class _HallucinationAggregator(ExampleMetric):
    @cached_property
    def _insertion_metrics(self) -> int:
        """Get the number of hallucinated medical term insertions for this example."""
        ref = self.example.ref.joined(normalized=True)
        hyp = self.example.hyp.joined(normalized=True)
        alignments = error_align(ref=ref, hyp=hyp)

        # Check if the alignment has deletions
        consecutive_insertions = 0
        has_contiguous_insertions = False
        insertions = 0
        for alignment in alignments:
            if alignment.op_type == OpType.INSERT:
                consecutive_insertions += 1
                insertions += 1
                if consecutive_insertions > self.src_metric.threshold:
                    has_contiguous_insertions = True
            else:
                consecutive_insertions = 0

        return dict(
            has_contiguous_insertions=has_contiguous_insertions,
            insertions=insertions,
        )

    @cached_property
    def has_contiguous_insertions(self) -> bool:
        """Check if there are contiguous hallucinated medical term insertions."""
        return self._insertion_metrics["has_contiguous_insertions"]

    @cached_property
    def insertions(self) -> int:
        """Get the number of hallucinated medical term insertions."""
        return self._insertion_metrics["insertions"]


@METRIC_REGISTRY.register("_legacy_hlcn")
class HallucinationAggregator(Metric):
    short_name = "Hallucination Insertions"
    long_name = "Hallucination Insertions"
    description = "Number of insertions that appear in "
    example_cls = _HallucinationAggregator

    def __init__(
        self,
        src: "Dataset",
        name: str = "_legacy_hlcn",
        threshold: int = 2,
    ):
        self.threshold = threshold
        super().__init__(src, name)


@METRIC_REGISTRY.register("legacy_deletions")
class Insertions(Metric):
    short_name = "Hallucination Insertions"
    long_name = "Hallucination Insertions"
    description = "Average number of insertions per example."

    @cached_property
    def value(self) -> int:
        """Get the number of hallucinated medical term insertions."""

        def get_insertions(example: "Example") -> int:
            return int(example.metrics._legacy_hlcn.insertions)

        return sum([get_insertions(example) for example in self._src_dataset]) / len(self._src_dataset)


@METRIC_REGISTRY.register("legacy_del_hallucinations")
class DeletionHallucinations(Metric):
    short_name = "Insertion Hallucinations"
    long_name = "Insertion Hallucinations"
    description = "Fraction of examples for which more than N consecutive insertions are observed."

    @cached_property
    def value(self) -> int:
        """Get the number of examples with hallucinated deletions."""

        def get_int_value(example: "Example") -> int:
            return int(example.metrics._legacy_hlcn.has_contiguous_insertions)

        return sum([get_int_value(example) for example in self._src_dataset]) / len(self._src_dataset)
