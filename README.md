# Talk to MOTHER

A simple web page where you can talk to MOTHER, a small language model
trained from scratch, with no code. Type a question, she answers. Every
answer gets a thumbs up / thumbs down: thumbs up is one tap, done,
and thumbs down asks what she should have said. The ratings and
reasons are stored and feed MOTHER's next training round.

## The design

The page is built to be the simplest, warmest chat page possible — not
a clone of any AI app. One thread, like messaging a person: your
messages in warm dark bubbles on the right, hers in white cards on the
left with her "M" mark. The empty state greets you and offers tappable
starter questions she can actually answer well (verified on the S11
champion: 312 + 4, after 549, saxophone, falcon), so you see what she can do
instead of guessing. Rating is one tap: thumbs up saves immediately,
done. Thumbs down opens a "why" field asking what she should have
said — that correction is what trains her.
Typing indicator, auto-scroll, and a floating rounded composer round
it out. Single self-contained page, no build step, no new dependencies.

One honest limit: the conversation is in the UI only. Her training
never included multi-turn dialogue, so every question is answered
fresh with the verified single-turn call (prompt
`Lesson: {question}\nExplanation:`, greedy, 48 tokens) — chat history
is deliberately NOT fed into the prompt, because that would break the
verified behavior. Teaching her real multi-turn conversation is a
future training experiment, not a UI change.

## What MOTHER is

MOTHER is a 29.6-million-parameter transformer (8 layers, 512 wide,
8,000-word vocabulary), trained in stages on Kaggle. This app runs the
S11 champion, the best version so far. Inference is pure numpy, no
torch, so the app is light enough for Render's free tier. Talking to her does NOT train
her — chatting never changes her weights. The feedback you leave here
becomes lessons and corrections in the next training round.

## Deploy on Render

1. Add the model files to `models/` first (see `models/README.md`):
   the S11 weights as six chunks (`mother_v008_s11.npz.part00` through
   `.part05`, each under 25 MB) and the tokenizer `mother_v006.model`
   (single file). Upload them through the GitHub web UI (repo page →
   Add file → Upload files, into `models/`). The app will not answer
   questions until all seven files are in the repo.
2. In the Render dashboard: **New + → Web Service → Build and deploy
   from a Git repository** (or **New + → Blueprint**). Connect this repo
   (`Darius-Lea/mother-chat`), branch `main`.
3. Render picks up `render.yaml` automatically: Python 3.11, free plan,
   build reassembles the weights chunks, verifies their SHA-256, then
   `pip install -r requirements.txt`; start
   `gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app`.
4. Deploy. The first build takes a few minutes (numpy +
   sentencepiece are small downloads; there is no torch any more). The
   service gets a URL like `https://mother-chat.onrender.com`.
5. Open the URL and ask her something. If the page says "MOTHER is not
   ready", it names the missing file — add it and redeploy.

Free-tier note: the service sleeps after 15 minutes with no visitors,
so the first visit after a nap takes ~30 seconds to wake up. The $7/mo
Starter plan keeps it awake.

## How the feedback works

- Every question + answer is stored in SQLite (`feedback.db`).
- Thumbs up saves with one tap, no text. Thumbs down opens a "why"
  field asking what she should have said, and that reason is saved
  against the answer.
- **Export:** `/feedback/export` downloads everything as JSON
  (`?format=csv` for CSV). `/feedback` shows a simple table in the browser.
- **Training:** the ratings become lessons for the next training round.
  See `training/README.md` for the pipeline: thumbs up turns her answer
  into a positive lesson, thumbs down + why turns your reason into the
  correction, and the lessons live in `training/feedback-lessons.jsonl`.
  Muse does the export-then-commit step; the app itself cannot push to
  GitHub (it has no credentials on Render).
- **Important:** on Render's free tier the disk is ephemeral — the
  feedback database is wiped on every redeploy or restart. Export the
  feedback regularly (at least before each redeploy). A persistent disk
  or a Render Postgres database is the follow-up fix; see "Known limits".

## Internet search

When a question needs current or external facts (news, prices, what
happened today, a CVE), the app searches the web and feeds the top
results into MOTHER's prompt as context, so she answers grounded in
them. The chat page shows "used web search" under answers that used
it.

- **No API key.** `search.py` tries three HTML search backends in
  order (DuckDuckGo HTML, Bing HTML, DuckDuckGo Lite) and takes the
  first one that returns results. Standard library only, no new
  dependencies.
- **Silent failure.** If search fails, times out, or there is no
  internet, she just answers without it. The user never sees a
  search error.
- **Trigger.** A question searches only when it looks like it needs
  current facts: words like today, latest, news, price, weather, a
  year (2024-2026), a CVE id, or an explicit "search the web" /
  "look up". Ordinary questions (math, what things are, chat) never
  search.
- **Config:** `MOTHER_SEARCH_ENABLED` (default 1, set 0 to disable),
  `MOTHER_SEARCH_TIMEOUT` (seconds per backend, default 4),
  `MOTHER_SEARCH_MAX_RESULTS` (default 3). `/health` reports
  `search_enabled`.
- **Tests:** `python tests/test_search.py` (offline, 53 checks: the
  parsers, the trigger logic, the fallback chain, and the full
  `/ask` flow with a stubbed model). `python tests/test_search_live.py`
  hits the real backends and needs internet.

One honest limit: MOTHER was never trained with a Context line in
her prompt, so search grounding is best-effort until a training
round teaches her to use context. The results are in her window and
she conditions on them, but how faithfully she follows them is
measured, not assumed. The feedback ratings on searched answers are
the measurement.

## Swapping in a new champion

When a new training round produces a better champion (e.g. R15),
see `models/README.md` for the full swap procedure (split the new
weights into chunks, write the new `.sha256` file, update
`MOTHER_MODEL_PATH` and the build command in `render.yaml`).

The model code (`mother_model.py`) only changes when the architecture
changes — every V008 champion shares the same shape, so a newer V008
round only needs a new weights file (plus its stage/epoch/val-loss
constants). The loader checks the parameter count (29,578,240) on
startup and refuses to run a file with the wrong shape.

## How the weights file is stored (and why not Git LFS)

The 118 MB weights file is stored in this repo as six chunks
(`mother_v008_s11.npz.part00` through `.part05`, each under 25 MB),
reassembled at build time by the command in `render.yaml` (which also
verifies the SHA-256). The chunks exist because the file is over
GitHub's 100 MB per-file limit, GitHub's web upload caps at 25 MB per
file, and the Muse GitHub connection cannot push binary files; see
`models/README.md` for the chunk sizes and the swap procedure. We
deliberately do NOT use Git LFS:

- Render does fetch LFS files during its git clone, so LFS would work.
- But GitHub's free LFS quota is 1 GB storage / 1 GB bandwidth per
  month, and every Render deploy re-downloads the file, eating that
  bandwidth. A 118 MB file allows only ~8 deploys a month before LFS
  bandwidth runs out and deploys start failing.
- Direct chunks have no such quota. The cost is repo history bloat
  (each champion version stays in git history forever); that is
  acceptable while champions are ~118 MB and replaced one at a time.

## Run locally

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put mother_v008_s11.npz (reassemble the six chunks first) and mother_v006.model in models/
python app.py
# open http://127.0.0.1:5000
```

## Known limits (v1)

- **No login.** Anyone with the URL can chat and see/export feedback.
  Add simple auth (or a private service) before sharing the link widely.
- **Feedback is ephemeral** on the free tier (see above). Export often.
- **One worker.** The app runs a single gunicorn worker so the model is
  loaded once; answers are generated one at a time behind a lock. Fine
  for one user; it is not built for many simultaneous chatters.
- **CPU only.** Answers take a few seconds on Render's free instance.
- **Single-turn answers.** Each question is answered fresh; she does
  not remember earlier messages in the thread (see "The design").

## Pending

- [ ] Persistent feedback storage (disk or Postgres) + simple auth.

Live-tokenizer test done 2026-11-15: the app with the real S11
weights + real `mother_v006.model` reproduces the recorded S11
disk-verification generations exactly (3/3 questions, including the
known failure — 7+5→13 — which is the proof it is the real model
answering).

## Files

| File | What it is |
| --- | --- |
| `app.py` | Flask app: chat page, /ask, feedback, export |
| `mother_model.py` | MotherV008 numpy engine + greedy generation (pure numpy, no torch) |
| `search.py` | Internet search (no API key) + prompt-context builder |
| `tests/` | `test_search.py` (offline suite) + `test_search_live.py` (real backends) + fixtures |
| `templates/index.html` | The chat page (mobile-friendly) |
| `templates/feedback.html` | Simple feedback table |
| `requirements.txt` | numpy + sentencepiece + flask + gunicorn |
| `render.yaml` | Render blueprint (free web service) |
| `models/` | Weights chunks + tokenizer + SHA file (chunks/tokenizer uploaded separately; see `models/README.md`) |
| `training/` | Feedback-to-training pipeline: `feedback-lessons.jsonl` (lessons built from chat ratings) + `training/README.md` (the pipeline, in plain words) |
