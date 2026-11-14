"""M3 helpers: clip trimming, SaT word probabilities, back-check routing."""

from __future__ import annotations

import numpy as np

from dub.contracts.v1 import BoundaryReq, TrimReq
from dub.core.router import Router, parse_overrides
from dub.ops.audio_io import read_wav, write_wav
from dub.ops.media import trim
from dub.providers.boundaries_sat import build_text, calibrate, word_probs
from dub.store.cas import CAS


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
