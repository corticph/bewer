from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from bewer.alignment import OpType
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_property

from ._keyterm_params import KeyTermParams

if TYPE_CHECKING:
    from bewer.key_term.key_term import KeyTermMatch

__all__: list[str] = []


class BUWEROutcome(NamedTuple):
    """Per-example B-WER/U-WER statistics, following Le et al. (2021, arXiv:2104.02194).

    Every alignment operation is attributed exactly once: substitutions and deletions
    by their reference token, insertions by their hypothesis token. The
    denominator fields count every reference token exactly once. Consequently
    ``num_biased_edits + num_unbiased_edits`` equals plain WER's edit count and
    ``num_biased_ref_tokens + num_unbiased_ref_tokens`` equals the reference length.
    """

    num_biased_edits: int
    num_unbiased_edits: int
    num_biased_ref_tokens: int
    num_unbiased_ref_tokens: int


class _BUWEROutcomes_(ExampleMetric):
    @metric_property(is_value=True)
    def ref_matches(self) -> list[KeyTermMatch]:
        """Vocabulary term occurrences found in the reference text."""
        return self.parent_metric._matcher.find_matches(self.example.ref)

    @metric_property(is_value=True)
    def hyp_matches(self) -> list[KeyTermMatch]:
        """Vocabulary term occurrences found in the hypothesis text."""
        return self.parent_metric._matcher.find_matches(self.example.hyp)

    @metric_property
    def _ref_vocab_positions(self) -> set[int]:
        """Reference token positions covered by a vocabulary term occurrence."""
        return {pos for match in self.ref_matches for pos in range(match.start, match.stop)}

    @metric_property
    def _hyp_vocab_positions(self) -> set[int]:
        """Hypothesis token positions covered by a vocabulary term occurrence."""
        return {pos for match in self.hyp_matches for pos in range(match.start, match.stop)}

    @metric_property
    def _counts(self) -> BUWEROutcome:
        """Attribute every alignment operation to the biased or the unbiased side.

        Substitutions and deletions are attributed by their reference token,
        insertions by their hypothesis token (Le et al., 2021). A token is
        biased only when it is part of a matched vocabulary term occurrence.
        """
        alignment = self.parent_metric._alignment[self.example.index].alignment
        num_biased_edits = 0
        num_unbiased_edits = 0
        num_biased_ref_tokens = 0
        num_unbiased_ref_tokens = 0
        for op in alignment:
            if op.type == OpType.INSERT:
                # Levenshtein operations tag their hypothesis token directly,
                # so insertion attribution is O(1) per operation.
                biased = op.hyp_token_idx in self._hyp_vocab_positions
                if biased:
                    num_biased_edits += 1
                else:
                    num_unbiased_edits += 1
                continue
            biased = op.ref_token_idx in self._ref_vocab_positions
            if op.type == OpType.MATCH:
                if biased:
                    num_biased_ref_tokens += 1
                else:
                    num_unbiased_ref_tokens += 1
                continue
            # SUBSTITUTE / DELETE: reference-side edits.
            if biased:
                num_biased_edits += 1
                num_biased_ref_tokens += 1
            else:
                num_unbiased_edits += 1
                num_unbiased_ref_tokens += 1
        return BUWEROutcome(num_biased_edits, num_unbiased_edits, num_biased_ref_tokens, num_unbiased_ref_tokens)

    @metric_property(is_value=True)
    def num_biased_edits(self) -> int:
        """Edits attributed to the biased (vocabulary) side."""
        return self._counts.num_biased_edits

    @metric_property(is_value=True)
    def num_unbiased_edits(self) -> int:
        """Edits attributed to the unbiased (non-vocabulary) side."""
        return self._counts.num_unbiased_edits

    @metric_property(is_value=True)
    def num_biased_ref_tokens(self) -> int:
        """Reference tokens attributed to the biased (vocabulary) side."""
        return self._counts.num_biased_ref_tokens

    @metric_property(is_value=True)
    def num_unbiased_ref_tokens(self) -> int:
        """Reference tokens attributed to the unbiased (non-vocabulary) side."""
        return self._counts.num_unbiased_ref_tokens


@METRIC_REGISTRY.register("_bu_wer_outcomes")
class _BUWEROutcomes(Metric):
    short_name_base = "_BUWEROutcomes"
    long_name_base = "Biased/Unbiased WER Outcomes"
    description = (
        "Private metric that computes the shared biased/unbiased WER statistics "
        "(num_biased_edits, num_unbiased_edits, num_biased_ref_tokens, num_unbiased_ref_tokens) "
        "used by B-WER (b_wer) and U-WER (u_wer), following the B-WER/U-WER protocol of "
        "Le et al. (2021, arXiv:2104.02194). Every alignment operation is attributed exactly "
        "once: substitutions and deletions by reference-token membership in the vocabulary, "
        "insertions by hypothesis-token membership. B-WER and U-WER therefore form an exact "
        "two-way split of plain WER. Parameters are the lean KeyTermParams "
        "(vocab, normalized). The KTR family's allow_subset_matches does not "
        "apply, since the union of covered token positions is invariant under subset "
        "removal. Not intended for direct use."
    )
    example_cls = _BUWEROutcomes_

    param_schema = KeyTermParams

    @metric_property
    def _matcher(self):
        """Build (and cache) the vocabulary matcher for this metric's vocabulary + pipeline."""
        vocab = self.dataset.config.vocabularies[self.params.vocab]
        return vocab.build_matcher(self.dataset, normalized=self.params.normalized)

    @metric_property(is_dependency=True)
    def _alignment(self):
        """The shared Levenshtein alignment instance, as in `_kt_stats`."""
        return self.dataset.metrics.levenshtein(
            normalized=self.params.normalized,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_property(is_value=True)
    def num_biased_edits(self) -> int:
        """Total edits attributed to the biased (vocabulary) side."""
        return sum(em.num_biased_edits for em in self)

    @metric_property(is_value=True)
    def num_unbiased_edits(self) -> int:
        """Total edits attributed to the unbiased (non-vocabulary) side."""
        return sum(em.num_unbiased_edits for em in self)

    @metric_property(is_value=True)
    def num_biased_ref_tokens(self) -> int:
        """Total reference tokens attributed to the biased (vocabulary) side."""
        return sum(em.num_biased_ref_tokens for em in self)

    @metric_property(is_value=True)
    def num_unbiased_ref_tokens(self) -> int:
        """Total reference tokens attributed to the unbiased (non-vocabulary) side."""
        return sum(em.num_unbiased_ref_tokens for em in self)
