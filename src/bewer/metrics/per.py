"""Punctuation Error Rate (PER) metric.

Based on the metric introduced in "LibriSpeech-PC: Benchmark for Evaluation of
Punctuation and Capitalization Capabilities of End-to-End ASR Models" (Meister
et al., 2023, IEEE ASRU).

PER isolates punctuation prediction errors from word-level errors by:
1. Masking all punctuation tokens to a unified ``<PUNCT>`` sentinel.
2. Aligning the masked sequences via Levenshtein editops (so all punctuation
   marks are interchangeable during alignment — a comma in the reference aligns
   with a period in the hypothesis rather than being scored as delete+insert).
3. Checking exact punctuation identity at each aligned position.

  PER = (I_P + D_P + S_P) / (I_P + D_P + S_P + C_P)

where S_P, D_P, I_P, C_P are punctuation substitutions, deletions, insertions,
and correct predictions respectively.  D_P and I_P are derived from the formulas:

  D_P = N_{P_ref} - (S_P + C_P)
  I_P = N_{P_hyp} - (S_P + C_P)

The metric runs under the ``with_punctuation`` tokenizer, which emits each
punctuation character as a standalone token.  The ``punct_chars`` parameter
should be a subset of the characters the tokenizer emits as standalone tokens;
characters that the tokenizer does not separate (e.g. apostrophes in
contractions) cannot be counted even if listed here.

Note: the default ``punct_chars`` includes both Unicode punctuation (e.g.
"\u201c", "\u201d", "«", "»") and their ASCII normalized forms (e.g. '"',
"<", ">") so that the metric works under both ``normalized=True`` and
``normalized=False``.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein as RFLevenshtein

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value

__all__ = ["PER"]

_PUNCT_MASK = "<PUNCT>"


def _mask_punct(tokens: list[str], punct_set: frozenset[str]) -> list[str]:
    """Replace every token consisting solely of *punct_set* characters with a unified mask."""
    return [_PUNCT_MASK if all(c in punct_set for c in t) else t for t in tokens]


def _is_punct(token: str, punct_set: frozenset[str]) -> bool:
    """True when every character in *token* belongs to *punct_set*."""
    return bool(token) and all(c in punct_set for c in token)


class PER_(ExampleMetric):
    """Example-level Punctuation Error Rate."""

    @metric_value
    def _per_components(self) -> tuple[int, int, int, int, int, int]:
        """Compute and cache PER components.

        Returns ``(S_P, I_P, D_P, C_P, N_{P_ref}, N_{P_hyp})``.
        """
        if self.params.normalized:
            ref_tokens = self.example.ref.tokens.normalized
            hyp_tokens = self.example.hyp.tokens.normalized
        else:
            ref_tokens = self.example.ref.tokens.standardized
            hyp_tokens = self.example.hyp.tokens.standardized

        punct_set = frozenset(self.params.punct_chars)

        n_punct_ref = sum(1 for t in ref_tokens if _is_punct(t, punct_set))
        n_punct_hyp = sum(1 for t in hyp_tokens if _is_punct(t, punct_set))

        ref_masked = _mask_punct(ref_tokens, punct_set)
        hyp_masked = _mask_punct(hyp_tokens, punct_set)

        editops = RFLevenshtein.editops(ref_masked, hyp_masked).as_list()

        ref_edit_idxs = {op[1] for op in editops if op[0] != "insert"}
        hyp_edit_idxs = {op[2] for op in editops if op[0] != "delete"}
        match_ref_indices = sorted(set(range(len(ref_masked))) - ref_edit_idxs)
        match_hyp_indices = sorted(set(range(len(hyp_masked))) - hyp_edit_idxs)

        c_p = 0
        s_p = 0
        for ref_idx, hyp_idx in zip(match_ref_indices, match_hyp_indices):
            if ref_masked[ref_idx] == _PUNCT_MASK:
                if ref_tokens[ref_idx] == hyp_tokens[hyp_idx]:
                    c_p += 1
                else:
                    s_p += 1

        d_p = n_punct_ref - (s_p + c_p)
        i_p = n_punct_hyp - (s_p + c_p)

        return (s_p, i_p, d_p, c_p, n_punct_ref, n_punct_hyp)

    @metric_value
    def num_substitutions(self) -> int:
        """Number of punctuation substitutions (position correct, value wrong)."""
        return self._per_components[0]

    @metric_value
    def num_insertions(self) -> int:
        """Number of spurious punctuation tokens in the hypothesis."""
        return self._per_components[1]

    @metric_value
    def num_deletions(self) -> int:
        """Number of punctuation tokens missing from the hypothesis."""
        return self._per_components[2]

    @metric_value
    def num_correct(self) -> int:
        """Number of correctly predicted punctuation tokens."""
        return self._per_components[3]

    @metric_value
    def num_punct_ref(self) -> int:
        """Total punctuation tokens in the reference."""
        return self._per_components[4]

    @metric_value
    def num_punct_hyp(self) -> int:
        """Total punctuation tokens in the hypothesis."""
        return self._per_components[5]

    @metric_value
    def num_edits(self) -> int:
        """Total punctuation edits (substitutions + insertions + deletions)."""
        s_p, i_p, d_p, _, _, _ = self._per_components
        return s_p + i_p + d_p

    @metric_value(main=True)
    def value(self) -> float:
        """Punctuation error rate for this example."""
        s_p, i_p, d_p, c_p, _, _ = self._per_components
        denom = i_p + d_p + s_p + c_p
        if denom == 0:
            return 0.0
        return (i_p + d_p + s_p) / denom


@METRIC_REGISTRY.register("per", tokenizer="with_punctuation")
class PER(Metric):
    short_name_base = "PER"
    long_name_base = "Punctuation Error Rate"
    description = (
        "Punctuation Error Rate (PER) measures the accuracy of punctuation prediction by "
        "masking all punctuation tokens to a unified label, aligning the masked sequences via "
        "Levenshtein distance, and then checking exact punctuation identity at aligned positions. "
        "PER = (I_P + D_P + S_P) / (I_P + D_P + S_P + C_P), where S_P, D_P, I_P, C_P are "
        "punctuation substitutions, deletions, insertions, and correct predictions respectively. "
        "A lower PER indicates better punctuation prediction."
    )
    example_cls = PER_

    @dataclass
    class param_schema(MetricParams):
        punct_chars: tuple[str, ...] = (
            ".",
            ",",
            "!",
            "?",
            ";",
            ":",
            "/",
            "(",
            ")",
            "-",
            '"',
            "\u201c",
            "\u201d",
            "\u201e",
            "«",
            "»",
            "<",
            ">",
            "¡",
            "¿",
        )
        normalized: bool = True

    @metric_value
    def num_substitutions(self) -> int:
        """Total punctuation substitutions across all examples."""
        return sum(em.num_substitutions for em in self)

    @metric_value
    def num_insertions(self) -> int:
        """Total punctuation insertions across all examples."""
        return sum(em.num_insertions for em in self)

    @metric_value
    def num_deletions(self) -> int:
        """Total punctuation deletions across all examples."""
        return sum(em.num_deletions for em in self)

    @metric_value
    def num_correct(self) -> int:
        """Total correctly predicted punctuation tokens across all examples."""
        return sum(em.num_correct for em in self)

    @metric_value
    def num_punct_ref(self) -> int:
        """Total punctuation tokens in all reference texts."""
        return sum(em.num_punct_ref for em in self)

    @metric_value
    def num_punct_hyp(self) -> int:
        """Total punctuation tokens in all hypothesis texts."""
        return sum(em.num_punct_hyp for em in self)

    @metric_value
    def num_edits(self) -> int:
        """Total punctuation edits across all examples."""
        return sum(em.num_edits for em in self)

    @metric_value(main=True)
    def value(self) -> float:
        """Dataset-level punctuation error rate (micro-averaged)."""
        s_p = self.num_substitutions
        i_p = self.num_insertions
        d_p = self.num_deletions
        c_p = self.num_correct
        denom = i_p + d_p + s_p + c_p
        if denom == 0:
            return 0.0
        return (i_p + d_p + s_p) / denom
