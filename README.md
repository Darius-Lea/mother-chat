# Talk to MOTHER

A simple web page where you can talk to MOTHER, a small language model
trained from scratch, with no code. Type a question, she answers. Every
answer gets a thumbs up / thumbs down button plus a "why" field — why
it's right or why it's wrong. The ratings and reasons are stored and
feed MOTHER's next training round.

## What MOTHER is

MOTHER is a 5.3-million-parameter transformer (4 layers, 256 wide,
8,000-word vocabulary), trained in stages on Kaggle. This app runs the
R14 champion, the best version so far. Talking to her does NOT train
her — chatting never changes her weights. The feedback you leave here
becomes lessons and corrections in the next training round.

## Deploy on Render

1. Add the model files to `models/` first (see `models/README.md`):
   the R14 weights as three chunks (`stageR14_champion.pt.part00/01/02`,
   each under 25 MB) and the tokenizer `mother_v006.model` (single file).
   Upload them through the GitHub web UI (repo page → Add file → Upload
   files, into `models/`). The app will not answer questions until all
   four files are in the repo.
2. In the Render dashboard: **New + → Web Service → Build and deploy
   from a Git repository** (or **New + → Blueprint**). Connect this repo
   (`Darius-Lea/mother-chat`), branch `main`.
3. Render picks up `render.yaml` automatically: Python 3.11, free plan,
   build reassembles the weights chunks, verifies their SHA-256, then
   `pip install -r requirements.txt`; start
   `gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app`.
4. Deploy. The first build takes a few minutes (CPU torch is a large
   download). The service gets a URL like
   `https://mother-chat.onrender.com`.
5. Open the URL and ask her something. If the page says "MOTHER is not
   ready", it names the missing file — add it and redeploy.

Free-tier note: the service sleeps after 15 minutes with no visitors,
so the first visit after a nap takes ~30 seconds to wake up. The $7/mo
Starter plan keeps it awake.

## How the feedback works

- Every question + answer is stored in SQLite (`feedback.db`).
- Thumbs up/down + the "why" reason are saved against that answer.
- **Export:** `/feedback/export` downloads everything as JSON
  (`?format=csv` for CSV). `/feedback` shows a simple table in the browser.
- **Important:** on Render's free tier the disk is ephemeral — the
  feedback database is wiped on every redeploy or restart. Export the
  feedback regularly (at least before each redeploy). A persistent disk
  or a Render Postgres database is the follow-up fix; see "Known limits".

## Swapping in a new champion

When a new training round produces a better champion (e.g. R15),
see `models/README.md` for the full swap procedure (split the new
weights into chunks, write the new `.sha256` file, update
`MOTHER_MODEL_PATH` and the build command in `render.yaml`).

The model code (`mother_model.py`) never changes — every champion shares
the same architecture; only the weights file changes. The loader checks
the parameter count (5,338,624) on startup and refuses to run a file
with the wrong shape.

## How the weights file is stored (and why not Git LFS)

The 64 MB weights file is stored in this repo as three chunks
(`stageR14_champion.pt.part00/01/02`, each under 25 MB), reassembled
at build time by the command in `render.yaml` (which also verifies
the SHA-256). The chunks exist because GitHub's web upload caps at
25 MB per file and the Muse GitHub connection cannot push binary
files; see `models/README.md` for the chunk sizes and the swap
procedure. We deliberately do NOT use Git LFS:

- Render does fetch LFS files during its git clone, so LFS would work.
- But GitHub's free LFS quota is 1 GB storage / 1 GB bandwidth per
  month, and every Render deploy re-downloads the file, eating that
  bandwidth. A 64 MB file allows only ~15 deploys a month before LFS
  bandwidth runs out and deploys start failing.
- Direct chunks have no such quota. The cost is repo history bloat
  (each champion version stays in git history forever); that is
  acceptable while champions are 64 MB and replaced one at a time.

## Run locally

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put stageR14_champion.pt and mother_v006.model in models/
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
- **CPU only.** Answers take ~1–2 seconds on Render's free instance.

## Pending

- [ ] Upload the weights chunks + tokenizer to `models/` via the
      GitHub web UI (3 chunks + `mother_v006.model`; see
      `models/README.md`).
- [ ] Create the Render service from this repo (dashboard step).
- [ ] Persistent feedback storage (disk or Postgres) + simple auth.

Live-tokenizer test done 2026-11-14: the app with the real R14
weights + real `mother_v006.model` reproduces the recorded R14
disk-verification generations exactly (8/8 questions, including the
known failures — 7+5→13 and hut→HECK — which is the proof it is the
real model answering).

## Files

| File | What it is |
| --- | --- |
| `app.py` | Flask app: chat page, /ask, feedback, export |
| `mother_model.py` | MotherV0061 architecture + greedy generation (copied from the MOTHER notebook) |
| `templates/index.html` | The chat page (mobile-friendly) |
| `templates/feedback.html` | Simple feedback table |
| `requirements.txt` | CPU-only torch + sentencepiece + flask + gunicorn |
| `render.yaml` | Render blueprint (free web service) |
| `models/` | Weights chunks + tokenizer + SHA file (chunks/tokenizer uploaded separately; see `models/README.md`) |
