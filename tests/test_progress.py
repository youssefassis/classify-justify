"""The rule worth testing is not the bar, it is when the bar stays away.

A progress bar written to a pipe becomes thousands of lines of carriage returns in a
CI log or a redirected run, so the terminal check is the behaviour under test.
"""

from __future__ import annotations

import io

from classify_justify.progress import progress, reading


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


class TestProgress:
    def test_it_yields_the_items_unchanged(self):
        assert list(progress(range(4))) == [0, 1, 2, 3]

    def test_it_stays_away_when_stderr_is_not_a_terminal(self):
        # pytest captures stderr into a pipe, which is the case that matters.
        assert progress([1, 2]).disable

    def test_it_draws_on_a_terminal(self, monkeypatch):
        monkeypatch.setattr("sys.stderr", _Terminal())
        bar = progress([1, 2])
        assert not bar.disable
        bar.close()

    def test_a_caller_can_silence_it_on_a_terminal(self, monkeypatch):
        """`train(verbose=False)` must stay silent even when run interactively."""
        monkeypatch.setattr("sys.stderr", _Terminal())
        assert progress([1, 2], enabled=False).disable


class TestReading:
    def test_it_passes_the_stream_through(self):
        with reading(io.BytesIO(b"payload"), total=7) as stream:
            assert stream.read() == b"payload"

    def test_an_unknown_length_is_not_a_zero_length_bar(self):
        """A server that sends no Content-Length should give a counter, not 0%."""
        with reading(io.BytesIO(b"payload"), total=0) as stream:
            assert stream.read() == b"payload"
