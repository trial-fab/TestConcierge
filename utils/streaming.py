import time as time_module


class SmoothStreamer:

    def __init__(
        self,
        placeholder,
        *,
        min_chars: int = 1,
        max_lag: float = 0.015,  # Much faster: 15ms max lag
        initial_hold: float = 0.02,  # Quick initial render: 20ms
        word_threshold: int = 2,  # Render every 2 words
    ) -> None:
        self._placeholder = placeholder
        self._min_chars = min_chars
        self._max_lag = max_lag
        self._initial_hold = initial_hold
        self._word_threshold = word_threshold
        self._last_flush = time_module.monotonic()
        self._rendered = ""
        self._latest = ""
        self._started = False
        self._word_buffer = 0

    def update(self, text: str | None) -> None:
        if not text:
            return
        if text == self._latest:
            return

        self._latest = text
        now = time_module.monotonic()
        delta = text[len(self._rendered):]

        if not delta:
            return

        # Fast initial render - show something immediately
        if not self._started:
            elapsed = now - self._last_flush
            if len(text) >= self._min_chars or elapsed >= self._initial_hold:
                self._flush(text)
                self._started = True
            return

        # Count words in delta for word-by-word rendering
        new_words = self._count_words(delta)
        self._word_buffer += new_words

        should_render = (
            self._word_buffer >= self._word_threshold
            or len(delta) >= self._min_chars * 3  # Substantial chunk
            or (now - self._last_flush) >= self._max_lag
        )

        if should_render:
            self._flush(text)
            self._word_buffer = 0

    def finalize(self, final_text: str | None = None) -> None:
        text = final_text if final_text is not None else self._latest
        if not text:
            return
        if text != self._rendered:
            self._flush(text)
        self._started = True

    def _flush(self, text: str) -> None:
        self._placeholder.write(text)
        self._rendered = text
        self._last_flush = time_module.monotonic()

    @staticmethod
    def _count_words(text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        return len(stripped.split())