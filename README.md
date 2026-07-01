# D&D Story Reader

A local text-to-speech tool for reading Dungeons & Dragons adventure text, session notes, or narration boxes out loud — built for Apple Silicon using the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) voice model via Apple's [MLX](https://github.com/ml-explore/mlx) framework.

Paste in a scene description, room text, or NPC dialogue, pick a voice, and it renders a `.wav` file you can drop straight into a session — fully offline, no API keys, no cloud calls. Handy for DMs who want a consistent narrator voice for pre-written boxed text, or for pre-recording flavor text and NPC lines ahead of a session.

![platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

## Features

- **Fully local inference** — runs on-device via MLX, tuned for Apple Silicon (M1/M2/M3/M4). No internet needed after the first model download, so it works fine at the table.
- **Simple GUI** — no command line required once set up, so any DM can run it.
- **10 built-in voices** — a mix of US/UK, male/female presets, useful for giving different NPCs or narration a distinct voice.
- **Automatic text cleanup** — expands contractions, normalizes punctuation and symbols so the model doesn't choke on raw text pasted straight from a module or homebrew document.
- **Smart chunking** — splits long boxed text or read-aloud passages into sentence/clause-sized pieces for more stable generation.
- **Automatic retry logic** — if a chunk fails to synthesize (a known intermittent tensor-shape issue in this model), the app automatically retries with several padded/reformatted variants before giving up on that chunk.

## Requirements

- macOS on Apple Silicon (M-series chip) — MLX is Apple Silicon–only.
- Python 3.10+
- [`espeak-ng`](https://github.com/espeak-ng/espeak-ng) installed on the system (used by the phonemizer backend)

### Python dependencies

```bash
pip install mlx-audio numpy soundfile phonemizer
```

> `tkinter` ships with most standard Python installs. If it's missing, install Python via [python.org](https://www.python.org/downloads/) or `brew install python-tk`.

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/xeonification/D-DStoryReader.git
cd D-DStoryReader

# 2. Install espeak-ng (required by phonemizer)
brew install espeak-ng

# 3. Install Python dependencies
pip install mlx-audio numpy soundfile phonemizer

# 4. Run it
python Main.py
```

On first launch, the app will download the `mlx-community/Kokoro-82M-bf16` model weights from Hugging Face Hub. This can take a minute or two depending on your connection — the status label at the top of the window will show progress.

## Usage

1. Wait for the status bar to read **"✅ Apple Silicon Model Fully Loaded & Ready"**.
2. Paste in your read-aloud text, room description, or NPC dialogue.
3. Choose a voice — e.g. a deeper voice for a villain, a lighter one for boxed narration.
4. Optionally set an output filename (defaults to `pyl` if left blank).
5. Click **Generate & Play**.
6. The finished `.wav` file is saved to `~/Desktop/<filename>.wav`, and a confirmation dialog will appear. Queue it up for the table, or build a folder of pre-generated NPC lines ahead of a session.

### Available voices

| Voice | Accent | Gender | Suggested use |
|---|---|---|---|
| `bm_fable`, `bm_george`, `bm_lewis` | British | Male | Narrator, older NPCs, nobles |
| `bf_emma`, `bf_isabella` | British | Female | Narrator, NPCs |
| `am_michael`, `am_adam` | American | Male | NPCs, general narration |
| `af_heart`, `af_bella`, `af_nicole` | American | Female | NPCs, general narration |

Voice codes starting with `b` automatically use `en-gb` phonemization; codes starting with `a` use `en-us`.

## How it works (under the hood)

**Text sanitization** — Kokoro is sensitive to certain punctuation and contracted forms, so before synthesis the input text goes through a cleanup pass that:
- Strips stray quote/backtick characters
- Expands contractions (`don't` → `do not`, `can't` → `cannot`, etc.)
- Converts em/en dashes and parentheses into commas, treating them as pause points
- Spells out `%`, `&`, and `$` as words

**Chunking** — text is split into sentences (on `.`, `!`, `?`), and any sentence longer than 12 words is further split on commas/semicolons/colons — useful for the long, comma-heavy boxed text common in D&D modules. This keeps each chunk short enough for stable, consistent generation.

**Retry-on-failure** — some chunks intermittently trigger a `broadcast_shapes` error inside the model's tensor pipeline. Rather than failing outright, the app retries each failed chunk with up to 7 lightly modified variants (trailing ellipsis, leading whitespace, a leading semicolon, a mid-sentence break, etc.) before giving up and skipping that chunk. Skipped chunks are logged to the console.

**Assembly** — all successfully generated audio chunks are concatenated with NumPy and written out as a single 24kHz mono `.wav` file via `soundfile`.

## Known limitations

- macOS + Apple Silicon only (MLX has no Windows/Linux/Intel Mac support).
- The `broadcast_shapes` retry logic is a workaround for what appears to be an upstream quirk in `mlx-audio` / Kokoro — it's not a guaranteed fix, and pathological input (dense stat blocks, tables, unusual formatting) may still cause a chunk to be skipped entirely.
- Output path is hardcoded to `~/Desktop`.
- No playback controls — despite the button label, the app currently only generates and saves; it does not auto-play the result. Not ideal for reading live at the table without a manual play step after generation.
- No batching for multiple passages at once — one chunk of text in, one `.wav` out per run.

## License

MIT (or your license of choice — update this section before publishing).

## Credits

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — the underlying TTS model
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX inference wrapper
- [MLX](https://github.com/ml-explore/mlx) — Apple's array framework for Apple Silicon
