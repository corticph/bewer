"""Tests for bewer.metrics.levenshtein module."""

from bewer.alignment import OpType
from bewer.metrics.alignment.levenshtein import Levenshtein, Levenshtein_


class TestLevenshteinAlignment:
    """Tests for Levenshtein alignment correctness."""

    def test_perfect_match_alignment(self, empty_dataset):
        """Alignment should be all MATCH ops when ref == hyp."""
        empty_dataset.add("hello world", "hello world")
        example = empty_dataset[0]
        alignment = example.metrics.levenshtein().alignment
        assert len(alignment) == 2
        assert all(op.type == OpType.MATCH for op in alignment)

    def test_substitution_alignment(self, empty_dataset):
        """Single substitution should produce one SUBSTITUTE op."""
        empty_dataset.add("hello world", "hello earth")
        example = empty_dataset[0]
        alignment = example.metrics.levenshtein().alignment
        subs = [op for op in alignment if op.type == OpType.SUBSTITUTE]
        assert len(subs) == 1
        assert subs[0].ref == "world"
        assert subs[0].hyp == "earth"

    def test_deletion_alignment(self, empty_dataset):
        """Deleted words should produce DELETE ops."""
        empty_dataset.add("testing one two three", "testing one two")
        example = empty_dataset[0]
        alignment = example.metrics.levenshtein().alignment
        dels = [op for op in alignment if op.type == OpType.DELETE]
        assert len(dels) == 1
        assert dels[0].ref == "three"

    def test_insertion_alignment(self, empty_dataset):
        """Extra hypothesis words should produce INSERT ops."""
        empty_dataset.add("hello world", "hello beautiful world")
        example = empty_dataset[0]
        alignment = example.metrics.levenshtein().alignment
        ins = [op for op in alignment if op.type == OpType.INSERT]
        assert len(ins) == 1
        assert ins[0].hyp == "beautiful"


class TestLevenshteinMatchOpsCorrectness:
    """Regression tests for set-iteration-order bug in _get_ops.

    bewer's Levenshtein._get_ops pairs unmatched ref/hyp token indices using
     ``zip(match_ref_indices, match_hyp_indices)`` where both are Python ``set``
     objects.  Sets have no guaranteed iteration order, so for certain index
     values (a run of ~5+ deletions followed by a short matched tail) two
     adjacent MATCH ops can end up with their hyp indices swapped — e.g.
     MATCH(ref="week", hyp="end") and MATCH(ref="end", hyp="week").
    """

    def test_match_ops_have_matching_text(self, empty_dataset):
        """Every MATCH op must have ref.lower() == hyp.lower().

        This is the fundamental invariant that the set-iteration-order bug
        violates.  If any MATCH op pairs non-corresponding tokens, this test
        fails.
        """
        # 9 ref tokens, 7 deletions, 2 matches at the tail.
        # match_ref_indices = {7, 8}, match_hyp_indices = {0, 1}.
        # set({7, 8}) can iterate as [8, 7], swapping the pairing.
        empty_dataset.add("a b c d e f g week end", "week end")
        example = empty_dataset[0]
        alignment = example.metrics.levenshtein().alignment

        match_ops = [op for op in alignment if op.type == OpType.MATCH]
        assert len(match_ops) == 2

        for op in match_ops:
            assert op.ref is not None
            assert op.hyp is not None
            assert op.ref.lower() == op.hyp.lower(), f"MATCH op has mismatched ref/hyp: ref={op.ref!r}, hyp={op.hyp!r}"

    def test_match_ops_preserve_positional_order(self, empty_dataset):
        """MATCH ops should pair ref/hyp tokens in positional order.

        The k-th unmatched ref token must match the k-th unmatched hyp token.
        """
        empty_dataset.add("a b c d e f g week end", "week end")
        example = empty_dataset[0]
        alignment = example.metrics.levenshtein().alignment

        match_ops = [op for op in alignment if op.type == OpType.MATCH]
        assert len(match_ops) == 2

        # ref token 7 ("week") should match hyp token 0 ("week")
        assert match_ops[0].ref_token_idx == 7
        assert match_ops[0].hyp_token_idx == 0
        assert match_ops[0].ref == "week"
        assert match_ops[0].hyp == "week"

        # ref token 8 ("end") should match hyp token 1 ("end")
        assert match_ops[1].ref_token_idx == 8
        assert match_ops[1].hyp_token_idx == 1
        assert match_ops[1].ref == "end"
        assert match_ops[1].hyp == "end"

    def test_many_deletes_then_two_matches(self, empty_dataset):
        """Stress the set ordering with a larger run of deletions.

        With 17 ref tokens and 15 deletions, match_ref_indices = {15, 16}.
        set({15, 16}) can iterate as [16, 15], again swapping the pair.
        """
        ref_tokens = [f"w{i}" for i in range(15)] + ["alpha", "beta"]
        hyp = "alpha beta"
        empty_dataset.add(" ".join(ref_tokens), hyp)
        example = empty_dataset[0]
        alignment = example.metrics.levenshtein().alignment

        match_ops = [op for op in alignment if op.type == OpType.MATCH]
        assert len(match_ops) == 2

        for op in match_ops:
            assert op.ref is not None
            assert op.hyp is not None
            assert op.ref.lower() == op.hyp.lower(), f"MATCH op has mismatched ref/hyp: ref={op.ref!r}, hyp={op.hyp!r}"

    def test_deletes_then_three_matches(self, empty_dataset):
        """Three trailing matches after many deletes.

        match_ref_indices = {5, 6, 7}, match_hyp_indices = {0, 1, 2}.
        set({5, 6, 7}) can iterate as [8, 5, 6, 7] or other non-sorted orders.
        """
        empty_dataset.add("x y z x y z x week end now", "week end now")
        example = empty_dataset[0]
        alignment = example.metrics.levenshtein().alignment

        match_ops = [op for op in alignment if op.type == OpType.MATCH]
        assert len(match_ops) == 3

        for op in match_ops:
            assert op.ref is not None
            assert op.hyp is not None
            assert op.ref.lower() == op.hyp.lower(), f"MATCH op has mismatched ref/hyp: ref={op.ref!r}, hyp={op.hyp!r}"

    def test_match_text_invariant_various_inputs(self):
        """All MATCH ops must have ref.lower() == hyp.lower() across varied inputs."""
        from bewer.core.dataset import Dataset

        cases = [
            ("hello world", "hello world"),
            ("the quick brown fox", "the quick brown dog"),
            ("a b c d e f g h i j k l m week end", "week end"),
            ("one two three four five six seven eight nine ten", "one ten"),
            ("alpha beta gamma delta epsilon", "alpha gamma epsilon"),
            ("the patient has a history of hypertension", "patient has history hypertension"),
        ]
        for ref, hyp in cases:
            dataset = Dataset()
            dataset.add(ref, hyp)
            example = dataset[0]
            alignment = example.metrics.levenshtein().alignment

            for op in alignment:
                if op.type == OpType.MATCH:
                    assert op.ref is not None
                    assert op.hyp is not None
                    assert op.ref.lower() == op.hyp.lower(), (
                        f"MISMATCH for ref={ref!r} hyp={hyp!r}: MATCH op ref={op.ref!r} hyp={op.hyp!r}"
                    )


class TestLevenshteinMetricValues:
    """Tests for Levenshtein metric value counts."""

    def test_num_edits_perfect_match(self, dataset_perfect_match):
        """num_edits is 0 for perfect match."""
        example = dataset_perfect_match[0]
        lev = example.metrics.levenshtein()
        assert lev.num_edits == 0

    def test_num_edits_substitution(self, empty_dataset):
        """num_edits counts substitutions."""
        empty_dataset.add("hello world", "hello earth")
        example = empty_dataset[0]
        lev = example.metrics.levenshtein()
        assert lev.num_edits == 1
        assert lev.num_substitutions == 1

    def test_num_edits_deletion(self, empty_dataset):
        """num_edits counts deletions."""
        empty_dataset.add("testing one two three", "testing one two")
        example = empty_dataset[0]
        lev = example.metrics.levenshtein()
        assert lev.num_edits == 1
        assert lev.num_deletions == 1

    def test_num_edits_insertion(self, empty_dataset):
        """num_edits counts insertions."""
        empty_dataset.add("hello world", "hello beautiful world")
        example = empty_dataset[0]
        lev = example.metrics.levenshtein()
        assert lev.num_edits == 1
        assert lev.num_insertions == 1

    def test_num_matches(self, empty_dataset):
        """num_matches counts matched tokens."""
        empty_dataset.add("hello world", "hello world")
        example = empty_dataset[0]
        lev = example.metrics.levenshtein()
        assert lev.num_matches == 2

    def test_num_matches_with_deletes(self, empty_dataset):
        """num_matches excludes deleted/inserted tokens."""
        empty_dataset.add("a b c d e f g week end", "week end")
        example = empty_dataset[0]
        lev = example.metrics.levenshtein()
        assert lev.num_matches == 2
        assert lev.num_deletions == 7


class TestLevenshteinMetricAttributes:
    """Tests for Levenshtein metric class attributes."""

    def test_short_name(self, sample_dataset):
        """Test Levenshtein short_name."""
        lev = sample_dataset.metrics.levenshtein()
        assert lev.short_name_base == "Levenshtein"

    def test_long_name(self, sample_dataset):
        """Test Levenshtein long_name."""
        lev = sample_dataset.metrics.levenshtein()
        assert lev.long_name_base == "Levenshtein Alignment"

    def test_description(self, sample_dataset):
        """Test Levenshtein has description."""
        lev = sample_dataset.metrics.levenshtein()
        assert len(lev.description) > 0

    def test_example_cls(self, sample_dataset):
        """Test Levenshtein has example_cls set."""
        lev = sample_dataset.metrics.levenshtein()
        assert lev.example_cls == Levenshtein_

    def test_metric_values_main(self):
        """Test that alignment is the main metric."""
        values = Levenshtein_.metric_values()
        assert values["main"] == "alignment"

    def test_metric_values_other(self):
        """Test that counts are in other metrics."""
        values = Levenshtein_.metric_values()
        assert "num_substitutions" in values["other"]
        assert "num_insertions" in values["other"]
        assert "num_deletions" in values["other"]
        assert "num_edits" in values["other"]
        assert "num_matches" in values["other"]

    def test_dataset_metric_values_main(self):
        """Dataset-level Levenshtein has no main metric."""
        values = Levenshtein.metric_values()
        assert values["main"] is None

    def test_dataset_metric_values_other(self):
        """Dataset-level Levenshtein aggregates the example-level edit counts."""
        values = Levenshtein.metric_values()
        for name in ("num_substitutions", "num_insertions", "num_deletions", "num_edits", "num_matches"):
            assert name in values["other"]

    def test_example_metric_values(self):
        """Test Levenshtein_ metric_values."""
        values = Levenshtein_.metric_values()
        assert values["main"] == "alignment"
        assert "num_substitutions" in values["other"]

    def test_dataset_level_counts_sum_examples(self, empty_dataset):
        """Dataset-level counts equal the sum of example-level counts."""
        empty_dataset.add("hello world", "hello there world")
        empty_dataset.add("one two three", "one two")
        lev = empty_dataset.metrics.levenshtein()
        assert lev.num_substitutions == sum(em.num_substitutions for em in lev)
        assert lev.num_insertions == sum(em.num_insertions for em in lev)
        assert lev.num_deletions == sum(em.num_deletions for em in lev)
        assert lev.num_edits == sum(em.num_edits for em in lev)
        assert lev.num_matches == sum(em.num_matches for em in lev)

    def test_dataset_counts_are_consistent(self, empty_dataset):
        """num_edits is substitutions + insertions + deletions at the dataset level."""
        empty_dataset.add("hello world", "hello there world")
        lev = empty_dataset.metrics.levenshtein()
        assert lev.num_edits == lev.num_substitutions + lev.num_insertions + lev.num_deletions


class TestAlignmentInheritance:
    """Tests for the is_alignment flag and alignment resolution."""

    def test_is_alignment_flag(self):
        """Alignment metrics set is_alignment; other metrics default to False."""
        assert Levenshtein.is_alignment is True
        from bewer.metrics.alignment.error_align import ErrorAlign
        from bewer.metrics.general.mer import MER

        assert ErrorAlign.is_alignment is True
        assert MER.is_alignment is False

    def test_alignment_metric_resolves_to_itself(self, sample_dataset):
        """A metric with is_alignment=True resolves its alignment to itself."""
        lev = sample_dataset.metrics.levenshtein()
        assert lev.alignment is lev

    def test_dataset_alignment_property_on_dependent_metric(self, sample_dataset):
        """A metric with an alignment dependency exposes the alignment metric."""
        mer = sample_dataset.metrics.mer()
        assert mer.alignment.is_alignment
        assert mer.alignment is sample_dataset.metrics.levenshtein()

    def test_example_alignment_property_on_dependent_metric(self, sample_dataset):
        """The example-level alignment property returns the example's Alignment."""
        em = sample_dataset[0].metrics.mer()
        assert em.alignment is sample_dataset[0].metrics.levenshtein().alignment
        assert em.alignment.num_edits == sum(1 for op in em.alignment if op.type != OpType.MATCH)

    def test_alignment_is_transitive(self, sample_dataset):
        """Alignment passes through non-alignment dependencies (WIL -> WIP -> Levenshtein)."""
        wil = sample_dataset.metrics.wil()
        assert wil.alignment is sample_dataset.metrics.levenshtein()
        em = sample_dataset[0].metrics.wil()
        assert em.alignment is sample_dataset[0].metrics.levenshtein().alignment

    def test_metric_without_alignment_resolves_none(self, sample_dataset):
        """Metrics without an alignment dependency resolve alignment to None."""
        wer = sample_dataset.metrics.wer()
        assert wer.alignment is None
        assert sample_dataset[0].metrics.wer().alignment is None

    def test_insertion_rate_alignment_param_selects_backend(self, sample_dataset):
        """The alignment param of insertion_rate controls the resolved alignment metric."""
        ir = sample_dataset.metrics.insertion_rate()
        assert ir.alignment is sample_dataset.metrics.levenshtein()
        ir_ea = sample_dataset.metrics.insertion_rate(alignment="error_align")
        assert ir_ea.alignment is sample_dataset.metrics.error_align()

    def test_key_term_metrics_inherit_alignment(self):
        """Key-term metrics inherit alignment through their outcome dependencies."""
        from bewer.configs import get_config
        from bewer.core.dataset import Dataset
        from bewer.key_term import Vocabulary

        config = get_config("base").clone().add_vocabulary(Vocabulary(name="key_terms", terms=["quick"]))
        dataset = Dataset(config=config)
        dataset.add("the quick brown fox", "the quick brown dog")
        ktr = dataset.metrics.key_term_recall(vocab="key_terms")
        assert ktr.alignment.is_alignment
        assert ktr.alignment is dataset.metrics.levenshtein(tokenizer="key_term")
        ktr_em = dataset[0].metrics.key_term_recall(vocab="key_terms")
        assert ktr_em.alignment is dataset[0].metrics.levenshtein(tokenizer="key_term").alignment
        rktr = dataset.metrics.relaxed_key_term_recall(vocab="key_terms")
        assert rktr.alignment is dataset.metrics.error_align(tokenizer="key_term")
