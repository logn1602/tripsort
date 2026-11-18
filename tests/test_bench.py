from tripsort._bench import peak_rss_bytes


def test_peak_memory_is_reported() -> None:
    peak = peak_rss_bytes()
    assert peak is not None
    assert 10 * 1024 * 1024 < peak < 64 * 1024**3  # an interpreter with numpy loaded


def test_peak_never_decreases() -> None:
    before = peak_rss_bytes()
    block = bytearray(64 * 1024 * 1024)  # touch 64 MB
    block[::4096] = b"x" * len(block[::4096])
    after = peak_rss_bytes()
    assert before is not None
    assert after is not None
    assert after >= before
