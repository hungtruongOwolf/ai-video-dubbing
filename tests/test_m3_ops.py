"""M3 helpers: clip trimming, SaT word probabilities, back-check routing."""

from __future__ import annotations

import numpy as np

from dub.contracts.v1 import BoundaryReq, TrimReq
from dub.core.router import Router, parse_overrides
from dub.ops.audio_io import read_wav, write_wav
from dub.ops.media import trim
from dub.providers.boundaries_sat import build_text, calibrate, word_probs
from dub.store.cas import CAS
from tests.conftest import requires_ffmpeg


def test_trim_cuts_silence_around_the_voice(tmp_path):
    cas = CAS(tmp_path / "cas")
    sr = 24000
    t = np.arange(sr) / sr
    voice = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    clip = np.concatenate([np.zeros(sr // 4), voice, np.zeros(sr // 2)]).astype(np.float32)
    src = tmp_path / "clip.wav"
    write_wav(src, clip, sr)
    res = trim(TrimReq(audio_key=cas.put_file(src)), cas)
    assert 220 <= res.lead_ms <= 240 and 470 <= res.tail_ms <= 490  # 20 ms of air kept
    audio, _ = read_wav(cas.path(res.audio_key))
    assert abs(len(audio) / sr * 1000 - res.dur_ms) <= 1 and 1030 <= res.dur_ms <= 1050


def test_sat_text_and_word_probabilities():
    req = BoundaryReq(lang="en", words=["Hello", "there", "how", "are", "you"])
    text, ends = build_text(req)
    assert text == "Hello there how are you" and ends == [5, 11, 15, 19, 23]
    probs = np.zeros(len(text))
    probs[10] = 0.6  # last char of "there"
    probs[11] = 0.8  # the space after it counts too
    assert word_probs(probs, text, ends)[1] == 0.8
    # Thai: no joiner, the ASR's own spaces are kept.
    th = BoundaryReq(
        lang="th", words=["ab", "cd", "ef"], space_before=[False, True, False], joiner=""
    )
    assert build_text(th) == ("ab cdef", [2, 5, 7])


def test_calibration_maps_the_model_threshold_to_one_half():
    assert calibrate(0.25, 0.25) == 0.5
    assert calibrate(0.9, 0.25) > 0.9 and calibrate(0.05, 0.25) < 0.2
    assert calibrate(0.3, 0.25) > calibrate(0.2, 0.25)  # order is kept


def test_backcheck_uses_the_target_language_asr_chain(cfg):
    r = Router(cfg, overrides=parse_overrides([]))
    # vi has no chain for asr.backcheck: it borrows its asr.transcribe chain (PhoWhisper).
    tr = None
    try:
        tr = r.route("asr.backcheck", {"src_lang": "en", "tgt_lang": "vi"})
    except Exception as e:  # noqa: BLE001 - no GPU workers on the laptop
        assert "phowhisper_large_ct2" in str(e)
    assert tr is None
    fake = Router(cfg, overrides=parse_overrides(["asr.*=fake_asr"]))
    assert fake.route("asr.backcheck", {"tgt_lang": "vi"}).chain == ["fake_asr"]


def test_boundary_f1_aligns_different_texts():
    from dub.bench.seg import boundary_f1

    gold = ["Hello there.", "How are you today?", "I am fine."]
    # The ASR lost the punctuation, misheard a word, and our split is late by one word.
    pred = ["hello there how", "are you to day i am fine"]
    f = boundary_f1(pred, gold, "en", tol=1)
    assert f.n_gold == 2 and f.n_pred == 1 and f.hits == 1
    assert boundary_f1(pred, gold, "en", tol=0).hits == 0
    zh = boundary_f1(["你好。", "今天天气很好。"], ["你好。", "今天天气很好。"], "zh", tol=0)
    assert zh.f1 == 1.0


def test_numpy_resampling_keeps_speech_band_and_removes_aliases():
    from dub.ops.audio_io import resample_int

    def tone(f, sr, sec=1.0):
        t = np.arange(int(sr * sec)) / sr
        return np.sin(2 * np.pi * f * t).astype(np.float32)

    def amp(x, f, sr):
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / (len(x) / 4)
        return spec[int(round(f * len(x) / sr))]

    up = resample_int(tone(1000, 24000), 2, 1)  # 24 -> 48 kHz
    assert len(up) == 48000 and abs(amp(up, 1000, 48000) - 1.0) < 0.05
    down = resample_int(tone(1000, 48000) + tone(10000, 48000), 1, 3)  # 48 -> 16 kHz
    assert len(down) == 16000 and abs(amp(down, 1000, 16000) - 1.0) < 0.05
    assert amp(down, 6000, 16000) < 0.01  # 10 kHz would alias to 6 kHz without the filter


@requires_ffmpeg
def test_master_reaches_the_target_loudness_with_one_gain(tmp_path):
    import numpy as np

    from dub.contracts.v1 import MasterReq
    from dub.ops.audio_io import write_wav
    from dub.ops.mix import integrated_lufs, master
    from dub.store.cas import CAS

    cas = CAS(tmp_path / "cas")
    t = np.arange(48000 * 6) / 48000
    quiet = (0.02 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # ~ -37 LUFS
    src = tmp_path / "in.wav"
    write_wav(src, np.stack([quiet, quiet], axis=1), 48000)
    assert integrated_lufs(src) < -30
    res = master(MasterReq(audio_key=cas.put_file(src), loudness_lufs=-16.0), cas)
    assert abs(integrated_lufs(cas.path(res.audio_key)) + 16.0) < 1.0
    assert abs(res.dur_ms - 6000) < 100


def test_extra_boundaries_at_reference_commas_are_soft():
    from dub.bench.seg import boundary_f1

    gold = ["We meet at a college, for we meet in an hour of change.", "No man can grasp it."]
    pred = [
        "We meet at a college.",
        "For we meet in an hour.",
        "Of change.",
        "No man can grasp it.",
    ]
    f = boundary_f1(pred, gold, "en", tol=1)
    d = f.as_dict()
    assert f.hits == 1 and d["soft_fp"] == 1 and d["hard_fp"] == 1
    assert d["clause_p"] == round(2 / 3, 4)


def test_min_length_guard_drops_short_sentence_ends_unless_sure():
    from dub.providers.boundaries_sat import min_length_guard

    probs = [0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.9]
    raw = [0.01, 0.6, 0.01, 0.01, 0.6, 0.995, 0.01, 0.9]
    out = min_length_guard(probs, raw, min_words=3)
    # word 1 closes a 2-word sentence: dropped; word 4 closes "w2..w4" (3 words): kept;
    # word 5 would close 1 word but SaT is sure: kept; the last word always ends one
    assert [i for i, p in enumerate(out) if p >= 0.5] == [4, 5, 7]
    assert min_length_guard(probs, raw, 0) == probs
