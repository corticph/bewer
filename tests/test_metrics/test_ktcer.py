"""Tests for bewer.metrics.ktcer module."""

import pytest

from bewer import Dataset
from bewer.metrics.ktcer import KTCER, KTCER_


class TestKTCERExampleMetric:
    """Tests for KTCER_ (ExampleMetric) class."""

    def test_perfect_match(self):
        dataset = Dataset()
        dataset.add(ref="the patient has diabetes", hyp="the patient has diabetes", key_terms={"k": ["diabetes"]})
        example = dataset[0]
        assert example.metrics.ktcer(vocab="k").value == 0.0

    def test_single_char_error(self):
        # "diabetis" vs "diabetes": 1 edit, 8 ref chars → CER = 1/8
        dataset = Dataset()
        dataset.add(ref="the patient has diabetes", hyp="the patient has diabetis", key_terms={"k": ["diabetes"]})
        example = dataset[0]
        ktcer = example.metrics.ktcer(vocab="k")
        assert ktcer.num_char_edits == 1
        assert ktcer.ref_chars == 8
        assert ktcer.value == pytest.approx(1 / 8)

    def test_num_char_edits_and_ref_chars(self):
        dataset = Dataset()
        dataset.add(
            ref="patient has diabetes and asthma",
            hyp="patient has diabetis and astma",
            key_terms={"k": ["diabetes", "asthma"]},
        )
        example = dataset[0]
        ktcer = example.metrics.ktcer(vocab="k")
        assert ktcer.num_char_edits == 2
        assert ktcer.ref_chars == 8 + 6  # len("diabetes") + len("asthma")
        assert ktcer.value == pytest.approx(2 / 14)

    def test_complete_deletion(self):
        # key term fully absent from hypothesis
        dataset = Dataset()
        dataset.add(ref="patient has diabetes", hyp="patient has", key_terms={"k": ["diabetes"]})
        example = dataset[0]
        ktcer = example.metrics.ktcer(vocab="k")
        assert ktcer.num_char_edits == 8  # all 8 chars deleted
        assert ktcer.ref_chars == 8
        assert ktcer.value == pytest.approx(1.0)

    def test_normalized_false_uses_raw_tokens(self):
        # With normalized=False, casing is preserved; a case mismatch counts as an edit.
        dataset = Dataset()
        dataset.add(ref="patient has Diabetes", hyp="patient has diabetes", key_terms={"k": ["Diabetes"]})
        example = dataset[0]
        ktcer_normalized = example.metrics.ktcer(vocab="k", normalized=True)
        ktcer_raw = example.metrics.ktcer(vocab="k", normalized=False)
        assert ktcer_normalized.value == 0.0
        assert ktcer_raw.value > 0.0

    def test_partial_boundary_penalty_flows_through(self):
        # "blood" key term, hyp "bloodpressure": +1 from hyp_right_partial
        dataset = Dataset()
        dataset.add(
            ref="patient has blood pressure",
            hyp="patient has bloodpressure",
            key_terms={"k": ["blood"]},
        )
        example = dataset[0]
        ktcer = example.metrics.ktcer(vocab="k")
        assert ktcer.num_char_edits == 1
        assert ktcer.ref_chars == 5


class TestKTCERDatasetMetric:
    """Tests for KTCER (dataset-level Metric) class."""

    @pytest.fixture
    def mixed_dataset(self):
        dataset = Dataset()
        dataset.add(
            ref="patient has diabetes",
            hyp="patient has diabetes",
            key_terms={"k": ["diabetes"]},
        )
        dataset.add(
            ref="patient has asthma",
            hyp="patient has astma",
            key_terms={"k": ["asthma"]},
        )
        return dataset

    def test_num_char_edits_aggregates(self, mixed_dataset):
        ktcer = mixed_dataset.metrics.ktcer(vocab="k")
        expected = sum(ex.metrics.ktcer(vocab="k").num_char_edits for ex in mixed_dataset)
        assert ktcer.num_char_edits == expected

    def test_ref_chars_aggregates(self, mixed_dataset):
        ktcer = mixed_dataset.metrics.ktcer(vocab="k")
        expected = sum(ex.metrics.ktcer(vocab="k").ref_chars for ex in mixed_dataset)
        assert ktcer.ref_chars == expected

    def test_value(self, mixed_dataset):
        # diabetes: 0 edits / 8 chars; asthma→astma: 1 edit / 6 chars → 1/14
        ktcer = mixed_dataset.metrics.ktcer(vocab="k")
        assert ktcer.value == pytest.approx(1 / 14)

    def test_all_correct(self):
        dataset = Dataset()
        dataset.add(ref="has diabetes", hyp="has diabetes", key_terms={"k": ["diabetes"]})
        dataset.add(ref="has asthma", hyp="has asthma", key_terms={"k": ["asthma"]})
        assert dataset.metrics.ktcer(vocab="k").value == 0.0

    def test_empty_vocab_raises(self):
        dataset = Dataset()
        with pytest.raises(ValueError, match="not found in dataset key term vocabularies"):
            dataset.metrics.ktcer(vocab="k").value


class TestKTCERMetricAttributes:
    def test_short_name_base(self):
        assert KTCER.short_name_base == "KTCER"

    def test_long_name_base(self):
        assert KTCER.long_name_base == "Key Term Character Error Rate"

    def test_description(self):
        assert len(KTCER.description) > 0

    def test_example_cls(self):
        assert KTCER.example_cls == KTCER_

    def test_metric_values_main(self):
        assert KTCER.metric_values()["main"] == "value"

    def test_metric_values_other(self):
        values = KTCER.metric_values()
        assert "num_char_edits" in values["other"]
        assert "ref_chars" in values["other"]
