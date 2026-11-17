# M13: Emotion of a line from emotion2vec

M12 picked each line's reference clip by four prosodic numbers: pitch, pitch range, level and
voicing. Those cannot tell anger from joy at the same pitch, nor a tired line from a sad one.
M13 adds a learned emotion model and measures, in every job, whether the dub's emotion follows
the source's.

## The model

emotion2vec+ large (Ma et al. 2024, [arXiv 2312.15185](https://arxiv.org/abs/2312.15185)):
- a self-supervised speech model (data2vec style), fine-tuned on pseudo-labelled emotional
  speech in many languages;
- about 300M parameters, 16 kHz input;
- `emotion2vec/emotion2vec_plus_large` on Hugging Face, revision `6c303ba9`.

For each clip it gives an utterance embedding (the mean of its frame features) and the
probabilities of 9 classes: angry, disgusted, fearful, happy, neutral, other, sad, surprised,
unknown.

- **License.** The FunASR Model Open Source License 1.1. It allows use, copying, modification
  and sharing, as long as the source and the model name are kept, and it describes the model
  as "for reference and learning purposes".
- **Runtime.** FunASR (MIT) runs it, in its own env `emo` (`pod/envs/emo.txt`). FunASR's
  dependencies (librosa, numba, hydra and more) stay away from OmniVoice's and VieNeu's pins.
  torch comes from the shared uv cache, so the volume does not hold a second copy.
- **Op.** `audio.emotion` (`EmotionReq`: a list of clip, start, end) on its own worker. It
  takes ports in `PORT_ORDER` (`emo` 7107), so the other workers keep theirs.

## How it is used (`dub.ops.emotion`)

- **Centred embeddings.** Each speaker's line embeddings, minus that speaker's mean, made unit
  length. What is left is how a line differs from the speaker's usual delivery, not who speaks
  or in which language. The dub's lines are centred on the dub's own mean.
- **Picking a clip** (`voice.style_by`):
  - `prosody` is M12;
  - `emotion` uses the cosine distance of the centred embeddings;
  - `both` uses the mean of the two distances, each divided by its median over the speaker's
    (line, clip) pairs.
  In `emotion` and `both`, a pool clip replaces the usual reference only when it scores lower
  by 0.25 (median units). Retakes use the usual reference, as in M12.
- **Agreement, in every job** (`metrics.json`):
  - `emotion_cos`: the mean cosine between a line's centred source and dub embeddings. It is 0
    when the dub's emotion has nothing to do with the source's.
  - `emotion_label_match`: of the lines whose source emotion is clear (a class other than
    neutral at 0.5 or more), the share whose dub has the same top class.

## Measured

The drama, same translations, four runs on the pod (`--param dub.voice...`):

| | One reference | M12 prosody | Emotion | **Both (default)** |
|---|---:|---:|---:|---:|
| Lines voiced from a pool clip | 0 | 60 | 69 | 63 |
| Emotion follows the source (`emotion_cos`) | 0.12 | 0.17 | **0.26** | 0.24 |
| Same top emotion, 46 clear lines | 0.28 | 0.35 | 0.37 | 0.35 |
| Pitch follows the source (correlation) | -0.10 | **0.46** | 0.34 | 0.45 |
| Pitch spread, dub / source | 0.26 | **0.81** | 0.71 | 0.70 |
| QA passed | 0.966 | **0.978** | 0.933 | 0.966 |
| Back-check CER p95 | 0.14 | 0.14 | 0.21 | 0.14 |
| Voice similarity p50 | 0.89 | 0.81 | 0.83 | 0.84 |
| Dub silent while the actor speaks, during / after the line | | 77 / 42 | 84 / 43 | 84 / 42 |

The cartoon has too few lines to tell the modes apart. Its `emotion_cos` went 0.07 (one
reference) → 0.26 (prosody) → 0.33 (emotion, both).

Reading it:
- **Emotion alone** brings the most emotion agreement, but costs intelligibility. Three more
  lines failed QA (CER p95 0.14 → 0.21): it picked clips whose pace and pitch were far from
  the line's.
- **Both** keeps prosody's pitch agreement and adds about 40% emotion agreement. It fails no
  line that the one-reference run did not fail: its three failures are the known name lines
  and "Căng đau ạ.". The cost against prosody alone is one QA failure, which comes and goes
  between runs (a name line), and 13 more windows of 250 ms where the dub is silent while the
  actor speaks.
- **A caution.** The emotion agreement is measured with emotion2vec, the same model that
  picks the clips. The measure is taken on the dub, so a higher `emotion_cos` does mean the
  TTS carried the clip's emotion into the line, as emotion2vec hears it. But the share of
  lines with the same top emotion did not move (0.35), and no human has listened. A blind
  listening test would settle whether `both` beats `prosody`. Switch back with
  `--param dub.voice.style_by=prosody`.

## Not done

- A blind listening test of the modes (the listening page of M10's list).
- emotion2vec for the QA gate: a take whose emotion is far from its line's could be retaken.
- Frame-level emotion (`granularity="frame"`) for lines that change mood halfway.
