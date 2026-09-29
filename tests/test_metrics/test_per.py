"""Tests for bewer.metrics.per module."""

from bewer.metrics.per import PER, PER_


class TestPERExampleMetric:
    """Tests for PER_ (ExampleMetric) class."""

    def test_perfect_match(self, empty_dataset):
        """PER is 0 when all punctuation is correct."""
        empty_dataset.add("hello, world.", "hello, world.")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.value == 0.0
        assert per.num_correct == 2
        assert per.num_substitutions == 0
        assert per.num_deletions == 0
        assert per.num_insertions == 0

    def test_substitution(self, empty_dataset):
        """Comma replaced by period counts as one substitution."""
        empty_dataset.add("hello, world.", "hello. world.")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.num_substitutions == 1
        assert per.num_correct == 1
        assert per.num_deletions == 0
        assert per.num_insertions == 0
        assert per.value == 1 / 2

    def test_deletion(self, empty_dataset):
        """Missing comma counts as one deletion."""
        empty_dataset.add("hello, world.", "hello world.")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.num_deletions == 1
        assert per.num_correct == 1
        assert per.num_substitutions == 0
        assert per.num_insertions == 0
        assert per.value == 1 / 2

    def test_insertion(self, empty_dataset):
        """Extra comma in hypothesis counts as one insertion."""
        empty_dataset.add("hello world", "hello, world")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.num_insertions == 1
        assert per.num_correct == 0
        assert per.num_substitutions == 0
        assert per.num_deletions == 0
        assert per.value == 1.0

    def test_all_punctuation_deleted(self, empty_dataset):
        """All punctuation missing from hypothesis."""
        empty_dataset.add("hello, world.", "hello world")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.num_deletions == 2
        assert per.num_correct == 0
        assert per.value == 1.0

    def test_all_punctuation_inserted(self, empty_dataset):
        """All punctuation is spurious in hypothesis."""
        empty_dataset.add("hello world", "hello, world.")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.num_insertions == 2
        assert per.num_correct == 0
        assert per.value == 1.0

    def test_no_punctuation(self, empty_dataset):
        """PER is 0 when neither ref nor hyp has punctuation."""
        empty_dataset.add("hello world", "hello world")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.value == 0.0
        assert per.num_punct_ref == 0
        assert per.num_punct_hyp == 0

    def test_multiple_punctuation_types(self, empty_dataset):
        """Mix of correct, substituted, deleted, inserted punctuation."""
        # ref:  "hello, world! how are you?"
        # hyp:  "hello. world how are you;"
        # Tokenized (with_punctuation):
        #   ref: hello , world ! how are you ?
        #   hyp: hello . world how are you ;
        # Punctuation: ref has , ! ?  (3 tokens)
        #              hyp has . ;     (2 tokens)
        # Masked alignment:
        #   ref: hello <PUNCT> world <PUNCT> how are you <PUNCT>
        #   hyp: hello <PUNCT> world how are you <PUNCT>
        # Alignment: hello=hello, <PUNCT>=<PUNCT>(match), world=world,
        #   <PUNCT> deleted, how=how, are=are, you=you, <PUNCT>=<PUNCT>(match)
        # Matched <PUNCT> positions:
        #   (1,1): ref=,  hyp=.  → substitution (S_P)
        #   (6,5): ref=?  hyp=;  → substitution (S_P)  -- wait, need to trace carefully
        # Actually: ref has 7 tokens [hello, ,, world, !, how, are, you, ?] -- no, 8 tokens
        # Let me recount.
        # ref: "hello, world! how are you?" → tokens: hello , world ! how are you ?
        #   = 8 tokens, 3 punct: , ! ?
        # hyp: "hello. world how are you;" → tokens: hello . world how are you ;
        #   = 7 tokens, 2 punct: . ;
        # masked ref: hello <PUNCT> world <PUNCT> how are you <PUNCT>  (8)
        # masked hyp: hello <PUNCT> world how are you <PUNCT>  (7)
        # editops: delete ref[3] (<PUNCT>=! since hyp has no <PUNCT> between world and how)
        # Wait, let me think again.
        # ref: [hello, <PUNCT>, world, <PUNCT>, how, are, you, <PUNCT>]  (indices 0-7)
        # hyp: [hello, <PUNCT>, world, how, are, you, <PUNCT>]  (indices 0-6)
        # editops(ref, hyp):
        #   hello=hello (match)
        #   <PUNCT>=<PUNCT> (match)
        #   world=world (match)
        #   <PUNCT>(ref[3]) deleted → delete (ref_idx=3)
        #   how=how (match)
        #   are=are (match)
        #   you=you (match)
        #   <PUNCT>=<PUNCT> (match, ref[7] vs hyp[6])
        # Matched <PUNCT> pairs:
        #   (1,1): ref=, hyp=. → different → S_P
        #   (7,6): ref=? hyp=; → different → S_P
        # C_P = 0, S_P = 2
        # D_P = 3 - 2 = 1 (the ! was deleted)
        # I_P = 2 - 0 = 2? No, I_P = N_hyp_punct - (S_P + C_P) = 2 - (2+0) = 0
        # Wait: I_P = 2 - (2 + 0) = 0. But that means no insertions, which is correct.
        # D_P = 3 - (2 + 0) = 1. The ! was deleted.
        # PER = (0 + 1 + 2) / (0 + 1 + 2 + 0) = 3/3 = 1.0
        empty_dataset.add("hello, world! how are you?", "hello. world how are you;")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.num_substitutions == 2  # , → . and ? → ;
        assert per.num_deletions == 1  # ! deleted
        assert per.num_insertions == 0
        assert per.num_correct == 0
        assert per.num_punct_ref == 3
        assert per.num_punct_hyp == 2
        assert per.value == 1.0

    def test_num_edits(self, empty_dataset):
        """num_edits is the sum of substitutions, insertions, and deletions."""
        empty_dataset.add("hello, world.", "hello. world,")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.num_edits == per.num_substitutions + per.num_insertions + per.num_deletions

    def test_standardized_tokens(self, empty_dataset):
        """PER works with normalized=False (standardized tokens)."""
        empty_dataset.add("hello, world.", "hello, world.")
        example = empty_dataset[0]
        per = example.metrics.per(normalized=False)
        assert per.value == 0.0
        assert per.num_correct == 2

    def test_extended_punctuation_chars(self, empty_dataset):
        """Default punct_chars covers extended characters from the voice-commands list."""
        # ¡ and ¿ are in the default punct_chars
        empty_dataset.add("¡hola! ¿qué?", "¡hola! ¿qué?")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.value == 0.0
        assert per.num_correct == 4  # ¡ ! ¿ ?

    def test_quotation_marks(self, empty_dataset):
        """Curly quotation marks are counted as punctuation."""
        empty_dataset.add("\u201chello\u201d", "\u201chello\u201d")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.value == 0.0
        assert per.num_correct == 2  # " "

    def test_punct_sentinel_no_collision(self, empty_dataset):
        """A literal token '<PUNCT>' in the text is not treated as punctuation."""
        # The sentinel uses angle brackets which the tokenizer would not emit
        # as a standalone token, so this is a safety check.
        empty_dataset.add("hello world", "hello world")
        example = empty_dataset[0]
        per = example.metrics.per()
        assert per.num_punct_ref == 0
        assert per.num_punct_hyp == 0
        assert per.value == 0.0


class TestPERCustomPunctChars:
    """Tests for custom punct_chars parameter."""

    def test_custom_punct_chars(self, empty_dataset):
        """Custom punct_chars includes additional characters."""
        empty_dataset.add("hello- world", "hello- world")
        example = empty_dataset[0]
        per = example.metrics.per(punct_chars=(".", ",", "!", "?", ":", ";", "-"))
        assert per.value == 0.0
        assert per.num_correct == 1

    def test_custom_punct_chars_excludes_default(self, empty_dataset):
        """When punct_chars excludes a character, that token is not counted as punctuation."""
        # ref has comma and period, but punct_chars only includes period
        empty_dataset.add("hello, world.", "hello world.")
        example = empty_dataset[0]
        per = example.metrics.per(punct_chars=(".",))
        # Only . is punctuation. Comma is treated as a regular word.
        # ref tokens: hello , world .
        # hyp tokens: hello world .
        # Only . counts as punctuation. It's correct in both.
        assert per.num_punct_ref == 1
        assert per.num_correct == 1
        assert per.num_deletions == 0
        assert per.value == 0.0


class TestPERDatasetMetric:
    """Tests for PER (dataset-level Metric) class."""

    def test_value_perfect_match(self, empty_dataset):
        """Dataset-level PER is 0 for perfect matches."""
        empty_dataset.add("hello, world.", "hello, world.")
        empty_dataset.add("goodbye, world.", "goodbye, world.")
        per = empty_dataset.metrics.per()
        assert per.value == 0.0

    def test_aggregates_across_examples(self, empty_dataset):
        """Dataset-level PER aggregates counts across examples."""
        # Example 1: perfect (C_P=2)
        empty_dataset.add("hello, world.", "hello, world.")
        # Example 2: one substitution (S_P=1, C_P=1)
        empty_dataset.add("foo, bar.", "foo. bar.")
        per = empty_dataset.metrics.per()
        assert per.num_correct == 3
        assert per.num_substitutions == 1
        assert per.num_deletions == 0
        assert per.num_insertions == 0
        # PER = 1 / (1 + 0 + 0 + 3) = 1/4
        assert per.value == 1 / 4

    def test_empty_dataset(self, empty_dataset):
        """PER on empty dataset returns 0."""
        per = empty_dataset.metrics.per()
        assert per.value == 0.0
        assert per.num_correct == 0
        assert per.num_edits == 0


class TestPERMetricAttributes:
    """Tests for PER metric class attributes."""

    def test_short_name(self, sample_dataset):
        per = sample_dataset.metrics.per()
        assert per.short_name_base == "PER"

    def test_long_name(self, sample_dataset):
        per = sample_dataset.metrics.per()
        assert per.long_name_base == "Punctuation Error Rate"

    def test_description(self, sample_dataset):
        per = sample_dataset.metrics.per()
        assert len(per.description) > 0

    def test_example_cls(self, sample_dataset):
        per = sample_dataset.metrics.per()
        assert per.example_cls == PER_


class TestPERMetricValues:
    """Tests for PER metric_values method."""

    def test_example_metric_values_main(self):
        values = PER_.metric_values()
        assert values["main"] == "value"

    def test_example_metric_values_other(self):
        values = PER_.metric_values()
        for name in ["num_substitutions", "num_insertions", "num_deletions", "num_correct", "num_edits"]:
            assert name in values["other"]

    def test_dataset_metric_values_main(self):
        values = PER.metric_values()
        assert values["main"] == "value"

    def test_dataset_metric_values_other(self):
        values = PER.metric_values()
        for name in ["num_substitutions", "num_insertions", "num_deletions", "num_correct", "num_edits"]:
            assert name in values["other"]
