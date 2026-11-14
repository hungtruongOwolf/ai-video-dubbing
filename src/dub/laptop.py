"""Laptop side: the video never leaves your machine.

1. `extract_audio`: copy the audio track out of the video without re-encoding
   (`ffmpeg -vn -c:a copy`, about 60-90 MB per hour of video). Only this is processed.
2. The pipeline returns the dubbed audio + subtitles.
3. `mux`: put them back into the original video without re-encoding the picture
   (`-c:v copy`, takes seconds). The original audio is kept as a second track.
"""

from __future__ import annotations

from pathlib import Path

from dub.ops.media import FFmpegError, probe, run_ffmpeg

ISO639_2 = {
    "ar": "ara",
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "hi": "hin",
    "id": "ind",
    "it": "ita",
    "ja": "jpn",
    "ko": "kor",
    "pt": "por",
    "ru": "rus",
    "th": "tha",
    "vi": "vie",
    "zh": "zho",
}


def lang3(lang: str) -> str:
    return ISO639_2.get(lang.split("-")[0].lower(), "und")


MP4_FAMILY = {".mp4", ".m4v", ".mov", ".m4a", ".3gp"}
EXTRACT_VERSION = 2  # 2: MP4 sources keep their container (see extract_audio)


def extract_audio(video: Path, out: Path) -> Path:
    """Copy the first audio track into its own file; returns the path written.

    MP4/MOV sources go into an .m4a, others into an .mka. The container matters for sync: an
    AAC track starts with ~1024 "priming" samples that MP4 hides with an edit list. Copied
    into Matroska that list is lost, and the whole timeline came out 21 ms late (measured).
    """
    info = probe(video)
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        raise FFmpegError(f"{video} has no audio track")
    # bitexact + no metadata: the same video always gives byte-identical audio, so the same
    # CAS key, the same job id and a 100% cache hit on a rerun.
    common = ["-map", "0:a:0", "-vn", "-sn", "-dn", "-c:a", "copy"]
    common += ["-map_metadata", "-1", "-fflags", "+bitexact"]
    if video.suffix.lower() in MP4_FAMILY:
        m4a = out.with_suffix(".m4a")
        try:
            run_ffmpeg(["-i", str(video), *common, "-f", "mp4", str(m4a)])
            return m4a
        except FFmpegError:
            m4a.unlink(missing_ok=True)  # a codec MP4 cannot hold: fall back to Matroska
    mka = out.with_suffix(".mka")
    run_ffmpeg(["-i", str(video), *common, str(mka)])
    return mka


def output_container(video: Path) -> str:
    """Keep MKV/WebM sources in MKV (any codec fits); everything else becomes MP4."""
    return ".mkv" if video.suffix.lower() in {".mkv", ".webm"} else ".mp4"


def mux(
    video: Path,
    dub_audio: Path,
    out: Path,
    *,
    tgt_lang: str,
    src_lang: str,
    subs: list[tuple[Path, str]],
) -> Path:
    """subs: [(srt path, language)], the first one is the default track."""
    has_video = any(s.get("codec_type") == "video" for s in probe(video).get("streams", []))
    mp4 = out.suffix.lower() == ".mp4"
    args = ["-i", str(video), "-i", str(dub_audio)]
    for path, _ in subs:
        args += ["-i", str(path)]
    if has_video:
        args += ["-map", "0:v:0"]
    args += ["-map", "1:a:0", "-map", "0:a:0?"]
    for i in range(len(subs)):
        args += ["-map", str(i + 2)]
    # The dub track is already AAC when it comes from the pipeline (.m4a): copy it as is.
    dub_codec = "copy" if dub_audio.suffix.lower() in (".m4a", ".aac") else "aac"
    args += ["-c:v", "copy", "-c:a:0", dub_codec, "-c:a:1", "aac", "-b:a", "192k"]
    if subs:
        args += ["-c:s", "mov_text" if mp4 else "srt"]
    args += [
        "-metadata:s:a:0",
        f"language={lang3(tgt_lang)}",
        "-metadata:s:a:0",
        f"title=Dub ({tgt_lang})",
        "-metadata:s:a:1",
        f"language={lang3(src_lang)}",
        "-metadata:s:a:1",
        "title=Original",
        "-disposition:a:0",
        "default",
        "-disposition:a:1",
        "0",
    ]
    for i, (_, lang) in enumerate(subs):
        args += [f"-metadata:s:s:{i}", f"language={lang3(lang)}"]
        args += [f"-disposition:s:{i}", "default" if i == 0 else "0"]
    if mp4:
        args += ["-movflags", "+faststart"]
    args += [str(out)]
    run_ffmpeg(args)
    return out
