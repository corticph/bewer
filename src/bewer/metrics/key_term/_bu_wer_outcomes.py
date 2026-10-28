from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NamedTuple

from bewer.alignment import OpType
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_property

from ._bu_wer_outcomes_params import BUWEROutcomesParams

if TYPE_CHECKING:
    from bewer.alignment.op import Op
    from bewer.key_term.key_term import KeyTermMatch

__all__: list[str] = []


class BUWEROutcome(NamedTuple):
    """Per-example B-WER/U-WER statistics, following Le et al. (2021, arXiv:2104.02194).

    Every alignment operation is attributed exactly once — substitutions and deletions
    by their reference token, insertions by their hypothesis token — and the
    denominator fields count every reference token exactly once. Consequently
    ``num_biased_errors + num_unbiased_errors`` equals plain WER's edit count and
    ``num_ref_vocab_tokens + num_ref_other_tokens`` equals the reference length.
    """

    num_biased_errors: int
    num_unbiased_errors: int
    num_ref_vocab_tokens: int
    num_ref_other_tokens: int


class _BUWEROutcomes_(ExampleMetric):
    @metric_property(is_value=True)
    def ref_matches(self) -> list[KeyTermMatch]:
        """Vocabulary term occurrences found in the reference text."""
        return self.parent_metric._matcher.find_matches(self.example.ref)

    @metric_property(is_value=True)
    def hyp_matches(self) -> list[KeyTermMatch]:
        """Vocabulary term occurrences found in the hypothesis text."""
        return self.parent_metric._matcher.find_matches(self.example.hyp)

    @metric_property(is_value=True)
    def _ref_vocab_positions(self) -> set[int]:
        """Reference token positions covered by a vocabulary term occurrence."""
        return {pos for match in self.ref_matches for pos in range(match.start, match.stop)}

    @metric_property(is_value=True)
    def _hyp_vocab_positions(self) -> set[int]:
        """Hypothesis token positions covered by a vocabulary term occurrence."""
        return {pos for match in self.hyp_matches for pos in range(match.start, match.stop)}

    def _insert_positions(self, op: "Op") -> set[int]:
        """Hypothesis token positions covered by an INSERT operation.

        The Levenshtein backend tags its operations with ``hyp_token_idx`` directly.
        The error_align backend does not, so positions are recovered from the
        operation's character span, which indexes the standardized hypothesis text —
        the same coordinate space as ``Token`` offsets.
        """
        if op.hyp_token_idx is not None:
            return {op.hyp_token_idx}
        positions: set[int] = set()
        span = op.hyp_span
        if span is None:
            return positions
        for i, token in enumerate(self.example.hyp.tokens):
            if token.start < span.stop and span.start < token.end:
                positions.add(i)
        return positions

    @metric_property(is_value=True)
    def _counts(self) -> BUWEROutcome:
        """Attribute every alignment operation to the biased or the unbiased side.

        Substitutions and deletions are attributed by their reference token,
        insertions by their hypothesis token (Le et al., 2021). In
        ``multi_token='occurrence'/'flag'`` mode a token is biased only when it is
        part of a matched vocabulary term occurrence; in ``multi_token='split'``
        mode a token is biased when it appears as a token of any vocabulary term.
        """
        if self.params.multi_token == "flag":
            terms = self.parent_metric._multi_token_terms
            if terms:
                warnings.warn(
                    f"Vocabulary '{self.params.vocab}' contains multi-token term(s): {', '.join(terms)}. "
                    "Occurrence semantics are used for multi-token terms: tokens of such terms only "
                    "count as biased when they are part of a matched term occurrence. Use "
                    "multi_token='split' for word-list semantics instead.",
                    category=UserWarning,
                )
        split_mode = self.params.multi_token == "split"
        vocab_tokens = self.parent_metric._vocab_tokens if split_mode else None

        alignment = self.parent_metric._alignment[self.example.index].alignment
        num_biased_errors = 0
        num_unbiased_errors = 0
        num_ref_vocab_tokens = 0
        num_ref_other_tokens = 0
        for op in alignment:
            if op.type == OpType.INSERT:
                if split_mode:
                    biased = op.hyp in vocab_tokens
                else:
                    positions = self._insert_positions(op)
                    biased = any(pos in self._hyp_vocab_positions for pos in positions)
                if biased:
                    num_biased_errors += 1
                else:
                    num_unbiased_errors += 1
                continue
            if split_mode:
                biased = op.ref in vocab_tokens
            else:
                biased = op.ref_token_idx in self._ref_vocab_positions
            if op.type == OpType.MATCH:
                if biased:
                    num_ref_vocab_tokens += 1
                else:
                    num_ref_other_tokens += 1
                continue
            # SUBSTITUTE / DELETE: reference-side errors.
            if biased:
                num_biased_errors += 1
                num_ref_vocab_tokens += 1
            else:
                num_unbiased_errors += 1
                num_ref_other_tokens += 1
        return BUWEROutcome(num_biased_errors, num_unbiased_errors, num_ref_vocab_tokens, num_ref_other_tokens)

    @metric_property(is_value=True)
    def num_biased_errors(self) -> int:
        """Errors attributed to the biased (vocabulary) side."""
        return self._counts.num_biased_errors

    @metric_property(is_value=True)
    def num_unbiased_errors(self) -> int:
        """Errors attributed to the unbiased (non-vocabulary) side."""
        return self._counts.num_unbiased_errors

    @metric_property(is_value=True)
    def num_ref_vocab_tokens(self) -> int:
        """Reference tokens attributed to the biased (vocabulary) side."""
        return self._counts.num_ref_vocab_tokens

    @metric_property(is_value=True)
    def num_ref_other_tokens(self) -> int:
        """Reference tokens attributed to the unbiased (non-vocabulary) side."""
        return self._counts.num_ref_other_tokens


@METRIC_REGISTRY.register("_bu_wer_outcomes")
class _BUWEROutcomes(Metric):
    short_name_base = "_BUWEROutcomes"
    long_name_base = "Biased/Unbiased WER Outcomes"
    description = (
        "Private metric that computes the shared biased/unbiased WER statistics "
        "(num_biased_errors, num_unbiased_errors, num_ref_vocab_tokens, num_ref_other_tokens) "
        "used by B-WER (b_wer) and U-WER (u_wer), following the B-WER/U-WER protocol of "
        "Le et al. (2021, arXiv:2104.02194): every alignment operation is attributed exactly "
        "once — substitutions and deletions by reference-token membership in the vocabulary, "
        "insertions by hypothesis-token membership — so B-WER and U-WER together form an exact "
        "two-way split of plain WER. Not intended for direct use."
    )
    example_cls = _BUWEROutcomes_

    param_schema = BUWEROutcomesParams

    @metric_property(is_value=False)
    def _matcher(self):
        """Build (and cache) the vocabulary matcher for this metric's vocabulary + pipeline."""
        vocab = self.dataset.config.vocabularies[self.params.vocab]
        return vocab.build_matcher(self.dataset, normalized=self.params.normalized)

    @metric_property(is_value=False)
    def _vocab_tokens(self) -> frozenset[str]:
        """The vocabulary's word list: every token of every term, individually."""
        tokens: set[str] = set()
        for key_term in self._matcher.key_terms:
            if self.params.normalized:
                tokens.update(key_term.tokens.normalized)
            else:
                tokens.update(key_term.tokens.standardized)
        return frozenset(tokens)

    @metric_property(is_value=False)
    def _multi_token_terms(self) -> tuple[str, ...]:
        """Vocabulary terms that tokenize to more than one token, sorted by raw text."""
        terms: list[str] = []
        for key_term in self._matcher.key_terms:
            token_list = key_term.tokens.normalized if self.params.normalized else key_term.tokens.standardized
            if len(token_list) > 1:
                terms.append(key_term.raw)
        return tuple(sorted(terms))

    @metric_property(is_dependency=True)
    def _alignment(self):
        """The shared alignment metric instance (levenshtein or error_align backend)."""
        if self.params.alignment == "error_align":
            return self.dataset.metrics.error_align(
                normalized=self.params.normalized,
                standardizer=self.standardizer,
                tokenizer=self.tokenizer,
                normalizer=self.normalizer,
            )
        return self.dataset.metrics.levenshtein(
            normalized=self.params.normalized,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_property(is_value=True)
    def num_biased_errors(self) -> int:
        """Total errors attributed to the biased (vocabulary) side."""
        return sum(em.num_biased_errors for em in self)

    @metric_property(is_value=True)
    def num_unbiased_errors(self) -> int:
        """Total errors attributed to the unbiased (non-vocabulary) side."""
        return sum(em.num_unbiased_errors for em in self)

    @metric_property(is_value=True)
    def num_ref_vocab_tokens(self) -> int:
        """Total reference tokens attributed to the biased (vocabulary) side."""
        return sum(em.num_ref_vocab_tokens for em in self)

    @metric_property(is_value=True)
    def num_ref_other_tokens(self) -> int:
        """Total reference tokens attributed to the unbiased (non-vocabulary) side."""
        return sum(em.num_ref_other_tokens for em in self)
