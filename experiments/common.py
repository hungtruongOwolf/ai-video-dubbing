"""Shared helpers for the experiments: test text (NTREX-128), ASR-like token streams, workers.

NTREX-128 (Microsoft, CC BY-SA 4.0): 1997 sentences of news, professionally translated into
128 languages, in 123 documents. The same documents in every language, so results compare
across languages.
"""

from __future__ import annotations

import json
import os
import unicodedata
import urllib.request
from pathlib import Path

from dub.contracts.v1 import Word

NTREX = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main"
FILES = {
    "en": "NTREX-128/newstest2019-src.eng.txt",
    "vi": "NTREX-128/newstest2019-ref.vie.txt",
    "th": "NTREX-128/newstest2019-ref.tha.txt",
    "zh": "NTREX-128/newstest2019-ref.zho-CN.txt",
    "ja": "NTREX-128/newstest2019-ref.jpn.txt",
    "es": "NTREX-128/newstest2019-ref.spa.txt",
}
LANGS = list(FILES)
CHAR_LANGS = {"zh", "ja"}  # Whisper gives one piece per character or two


def data_dir() -> Path:
    d = Path(os.environ.get("DUB_EXP_DATA", Path(__file__).parent / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(rel: str) -> Path:
    path = data_dir() / Path(rel).name
    if not path.exists():
        req = urllib.request.Request(f"{NTREX}/{rel}", headers={"User-Agent": "dub-experiments"})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    return path


def documents(lang: str) -> list[list[str]]:
    """NTREX sentences of `lang`, grouped by news document."""
    lines = _fetch(FILES[lang]).read_text(encoding="utf-8-sig").splitlines()
    ids = _fetch("DOCUMENT_IDS.tsv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(ids), (lang, len(lines), len(ids))
    docs: list[list[str]] = []
    prev = None
    for doc_id, s in zip(ids, lines, strict=True):
        if doc_id != prev:
            docs.append([])
            prev = doc_id
        docs[-1].append(" ".join(s.split()))
    return docs


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def strip_punct(s: str) -> str:
    return "".join(c for c in s if not _is_punct(c))


def asr_like(sentences: list[str], lang: str, punct: bool) -> tuple[list[str], list[bool]]:
    """Turn reference sentences into the token stream an ASR would give.

    punct=True: text as written (Whisper writes punctuation and capitals for most languages).
    punct=False: no punctuation, lowercase (PhoWhisper; many ASR models).
    zh/ja: one token per character, punctuation stuck to the character before it.
    th: dictionary words; `space_before` marks the spaces of the text (Thai separates phrases
    and sentences with spaces). Sentences are joined with a space, as Thai writes them.
    """
    tokens: list[str] = []
    space: list[bool] = []
    for s in sentences:
        if not punct:
            s = strip_punct(s).lower()
        if lang in CHAR_LANGS:
            for ch in s.replace(" ", ""):
                if _is_punct(ch) and tokens:
                    tokens[-1] += ch
                else:
                    tokens.append(ch)
                    space.append(False)
        elif lang == "th":
            from pythainlp.tokenize import word_tokenize

            pending_space = bool(tokens)  # sentences are separated by a space
            for w in word_tokenize(s, engine="newmm", keep_whitespace=True):
                if w.isspace():
                    pending_space = True
                    continue
                if _is_punct(w[0]) and tokens and not pending_space:
                    tokens[-1] += w
                    continue
                tokens.append(w)
                space.append(pending_space)
                pending_space = False
        else:
            for w in s.split():
                if w:
                    tokens.append(w)
                    space.append(True)
    return tokens, space


def sentences_from(
    tokens: list[str], space: list[bool], probs: list[float], thr: float, joiner: str
):
    """Cut the token stream where the boundary probability reaches `thr`."""
    out: list[list[Word]] = [[]]
    for t, sp, p in zip(tokens, space, probs, strict=True):
        out[-1].append(Word(w=t, start_ms=0, end_ms=0, space_before=sp))
        if p >= thr:
            out.append([])
    from dub.contracts.v1 import join_words

    return [join_words(ws, joiner) for ws in out if ws]


def joiner_for(lang: str) -> str:
    return "" if lang in ("zh", "ja", "th") else " "


# ---------------------------------------------------------------- pod workers


def worker_url(env: str) -> str:
    """URL of a worker started by dubd (ports as EnvManager assigns them)."""
    from dub.core.config import load_config
    from dub.pod.envs import worker_ports

    return f"http://127.0.0.1:{worker_ports(load_config())[env][1]}"


def run_worker(env: str, provider: str, op: str, request: dict, params: dict | None = None):
    from dub.core.config import load_config, worker_of
    from dub.worker.protocol import post_json

    cfg = load_config()
    env = worker_of(cfg.providers[provider]) or env  # a provider may have its own worker
    body = {
        "provider": provider,
        "op": op,
        "params": {**cfg.providers[provider].params, **(params or {})},
        "request": request,
    }
    return post_json(f"{worker_url(env)}/v1/run", body, timeout=600)["result"]


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
