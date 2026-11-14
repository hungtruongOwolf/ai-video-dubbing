"""Small CPU providers that need no model."""

from __future__ import annotations

from dub.contracts.v1 import Boundaries, BoundaryReq, Diarization, DiarizeReq, Turn
from dub.core.config import ProviderManifest
from dub.core.provider_api import ProviderError, RunCtx
from dub.ops.audio_io import duration_ms
from dub.ops.segment import rule_probs


class SingleSpeaker:
    """Diarization stand-in until M4: the whole file is one speaker, S1.

    Correct for monologues (talks, lectures, vlogs). Videos with several people get one cloned
    voice until real diarization lands.
    """

    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest

    def run(self, op: str, req: DiarizeReq, ctx: RunCtx) -> Diarization:
        if op != "audio.diarize":
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        end = duration_ms(ctx.cas.path(req.audio_key))
        return Diarization(turns=[Turn(speaker="S1", start_ms=0, end_ms=end)], speakers=["S1"])


class PunctRules:
    """Sentence boundaries from punctuation alone: the baseline SaT replaced (experiment E4).

    Abbreviations ("Mr.", "e.g.") and initials do not end a sentence. Useless on ASR output
    without punctuation; kept as the fallback when SaT is not available (laptop, --fake).
    """

    def __init__(self, manifest: ProviderManifest):
        self.manifest = manifest

    def run(self, op: str, req: BoundaryReq, ctx: RunCtx) -> Boundaries:
        if op != "text.boundaries":
            raise ProviderError(f"{self.manifest.id} does not implement {op}", kind="unsupported")
        return Boundaries(probs=rule_probs(req.words))
