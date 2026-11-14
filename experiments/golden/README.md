# Reference sentences for real videos (silver labels)

These files are the reference sentences used by `experiments/real_videos.py` for videos whose
subtitles have no sentence punctuation (Chinese and Japanese subtitles on Wikimedia Commons
separate phrases with spaces). One sentence per line.

- The words are the human subtitles of the first ~5 minutes of each video, unchanged (the
  script checks that joining the lines gives the transcript back).
- The sentence ends were placed by hand by Claude, the assistant that wrote this pipeline, at
  the end of statements, questions and speaker turns. They are not a human annotation, so
  call the resulting F1 a silver-label score.

| File | Video on Wikimedia Commons | Subtitles |
|---|---|---|
| `zh-vivo-interview.txt` | File:采访vivo & MediaTek研发大佬：蓝厂与天玑合作背后的故事.webm | TimedText …webm.zh.srt |
| `ja-digital-q.txt` | File:デジタル監から東京都副知事に３つの質問「都のDX」「ユーザー視点」「キャリア改革」.webm | TimedText …webm.ja.srt |
| `ja-mynaportal.txt` | File:初潜入。マイナポータル開発の舞台裏へ.webm | TimedText …webm.ja.srt |

Licences: see each file page on commons.wikimedia.org; Commons TimedText pages are under
CC BY-SA. The other videos (en, es, vi and one zh narration) have punctuated subtitles, and their
references are built from them at run time.

## YODAS2 videos (`experiments/yodas_videos.py`)

Same method for YODAS2 videos whose uploaded subtitles are only partly punctuated: the words
are the subtitles of the first ~5 minutes unchanged, the sentence ends placed by hand.

| File | YouTube video (via YODAS2 `vi000`/`zh000`) | Licence |
|---|---|---|
| `6_HyZDSeLVY.txt` | https://www.youtube.com/watch?v=6_HyZDSeLVY (Bali travel vlog, vi) | CC BY 3.0 (YODAS2: videos published under CC BY), subtitles by the uploader |
| `v5DZN7rsLh4.txt` | https://www.youtube.com/watch?v=v5DZN7rsLh4 (oil futures analysis, zh) | CC BY 3.0 (YODAS2), subtitles by the uploader |
| `t5IDTECWpWM.txt` | https://www.youtube.com/watch?v=t5IDTECWpWM (monthly zodiac talk, zh) | CC BY 3.0 (YODAS2), subtitles by the uploader |

Rule for the hand-placed ends, so the labels can be reproduced: an explicit `。！？` in the
subtitles always ends a sentence (a `；` between two independent clauses too); where the
subtitles have no mark (mostly at cue ends, also list items), a sentence ends when a new
statement starts.
