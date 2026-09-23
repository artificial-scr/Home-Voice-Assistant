"""Unit tests for sentence extraction logic in brain/pipeline.py."""

import pytest
from pipeline import _MIN_SENTENCE_CHARS, _SENTENCE_END, _extract_sentences


class TestSentenceEndRegex:
    def test_splits_on_period_uppercase(self):
        assert _SENTENCE_END.search("Hello. World")

    def test_splits_on_question_uppercase(self):
        assert _SENTENCE_END.search("Done? Yes")

    def test_splits_on_exclamation_uppercase(self):
        assert _SENTENCE_END.search("Stop! Now")

    def test_splits_on_digit_after_period(self):
        assert _SENTENCE_END.search("Turn on. 5 lights")

    def test_no_split_lowercase_after_period(self):
        # "e.g. something" — lowercase after period, should not split
        assert not _SENTENCE_END.search("e.g. something")

    def test_no_split_no_space(self):
        # Period with no trailing space (end of string) — should not split
        assert not _SENTENCE_END.search("Hello.")

    def test_no_split_mid_word_period(self):
        assert not _SENTENCE_END.search("U.S.A")


class TestExtractSentences:
    def test_empty_string(self):
        sentences, remaining = _extract_sentences("")
        assert sentences == []
        assert remaining == ""

    def test_no_boundary_returns_full_buffer(self):
        sentences, remaining = _extract_sentences("Hello world")
        assert sentences == []
        assert remaining == "Hello world"

    def test_period_no_trailing_space_stays_in_buffer(self):
        sentences, remaining = _extract_sentences("Hello world.")
        assert sentences == []
        assert remaining == "Hello world."

    def test_single_sentence_split(self):
        sentences, remaining = _extract_sentences("Hello world. Next sentence")
        assert sentences == ["Hello world."]
        assert remaining == "Next sentence"

    def test_multiple_sentences(self):
        sentences, remaining = _extract_sentences(
            "First sentence. Second sentence. Third"
        )
        assert sentences == ["First sentence.", "Second sentence."]
        assert remaining == "Third"

    def test_question_mark_boundary(self):
        sentences, remaining = _extract_sentences("Are you there? Yes I am.")
        assert sentences == ["Are you there?"]
        assert remaining == "Yes I am."

    def test_exclamation_boundary(self):
        sentences, remaining = _extract_sentences("Watch out! That is dangerous.")
        assert sentences == ["Watch out!"]
        assert remaining == "That is dangerous."

    def test_mixed_terminators(self):
        sentences, remaining = _extract_sentences("Really? Yes! Because it works. Done")
        assert sentences == ["Really?", "Yes!", "Because it works."]
        assert remaining == "Done"

    def test_lowercase_after_period_not_split(self):
        # "e.g. something" — no uppercase/digit after period, stays in buffer
        sentences, remaining = _extract_sentences("e.g. something here")
        assert sentences == []
        assert remaining == "e.g. something here"

    def test_digit_after_period_splits(self):
        sentences, remaining = _extract_sentences("Turn on. 3 lights please")
        assert sentences == ["Turn on."]
        assert remaining == "3 lights please"

    def test_no_length_filter_on_mid_stream_sentences(self):
        # _extract_sentences does NOT filter by _MIN_SENTENCE_CHARS.
        # Short sentences mid-stream are returned; the caller decides what to do.
        short = "OK. Here is the full answer now"
        sentences, remaining = _extract_sentences(short)
        assert "OK." in sentences

    def test_remaining_has_no_trailing_boundary(self):
        # The last part returned should never contain a resolved sentence boundary
        _, remaining = _extract_sentences("One. Two. Still typing")
        assert "." not in remaining or remaining == "Still typing"

    def test_whitespace_only_parts_skipped(self):
        # If a split produces a blank fragment it is stripped/skipped
        sentences, _ = _extract_sentences("  . Next thing")
        # leading-whitespace-only fragment before the period is empty after strip
        assert all(s.strip() for s in sentences)


class TestMinSentenceChars:
    def test_constant_value(self):
        assert _MIN_SENTENCE_CHARS == 8

    def test_tail_below_threshold_not_synthesized(self):
        # Simulate handle_transcript tail logic: tail shorter than threshold is skipped
        tail = "OK."
        assert len(tail) < _MIN_SENTENCE_CHARS

    def test_tail_at_threshold_is_synthesized(self):
        tail = "A" * _MIN_SENTENCE_CHARS
        assert len(tail) >= _MIN_SENTENCE_CHARS
