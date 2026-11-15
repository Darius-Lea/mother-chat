# models/

This folder holds the two files MOTHER needs to talk.

## What goes here

1. `mother_v008_s11.npz` — the V008 S11 champion weights, weights-only
   (118,342,428 bytes, SHA-256
   `48f539fe1a95a0ad91d8edf3d501721dcfaef0f159c6e5816505edcbf2f177ed`).
   This file is NOT committed directly: it is split into six chunks
   (see below) and reassembled at build time.
2. `mother_v006.model` — the SentencePiece tokenizer (368,413 bytes,
   SHA-256 `cd6ca7f8114951450fb48d6f65b2774013252cda34486ff1a63e3526ee1cee5d`),
   from the Kaggle dataset `mother-v0061-master-backup`. Committed as a
   single file.
3. `mother_v008_s11.npz.sha256` — the SHA-256 of the reassembled
   weights file. The build verifies it, so a corrupted chunk fails the
   deploy loudly instead of serving a broken model.

The older R14/R16/R17/R18 `.pt` chunks in this folder are left over
from earlier champions. The current build does not use them; they stay
in the repo (and in git history) as a record.

## Why the weights are split into chunks

The 118 MB weights file is over GitHub's 100 MB per-file limit, so a
direct commit is not allowed at all — and GitHub's web upload caps at
25 MB per file, while the Muse GitHub connection cannot push binary
files. So the weights live in the repo as six chunks, each under 25 MB:

| Chunk | Bytes |
| --- | --- |
| `mother_v008_s11.npz.part00` | 19,723,738 |
| `mother_v008_s11.npz.part01` | 19,723,738 |
| `mother_v008_s11.npz.part02` | 19,723,738 |
| `mother_v008_s11.npz.part03` | 19,723,738 |
| `mother_v008_s11.npz.part04` | 19,723,738 |
| `mother_v008_s11.npz.part05` | 19,723,738 |

The chunks were made with `split -b 19723738 -d -a 2
mother_v008_s11.npz mother_v008_s11.npz.part` and verified
byte-identical on reassembly (`cat` in order reproduces the original
SHA-256). The Render build command in `render.yaml` reassembles them,
checks the SHA-256, and only then installs dependencies:

```sh
cat models/mother_v008_s11.npz.part00 \
    models/mother_v008_s11.npz.part01 \
    models/mother_v008_s11.npz.part02 \
    models/mother_v008_s11.npz.part03 \
    models/mother_v008_s11.npz.part04 \
    models/mother_v008_s11.npz.part05 \
    > models/mother_v008_s11.npz \
  && sha256sum -c models/mother_v008_s11.npz.sha256 \
  && pip install -r requirements.txt
```

Upload the six chunks + the tokenizer + the `.sha256` file through
the GitHub web UI (repo page → Add file → Upload files, into
`models/`), or with git CLI from a machine with GitHub access. One
quirk: the web upload rejects a batch totalling ~59 MB or more even
though each file is under 25 MB, so upload one or two files per
commit. The app will start without them but the chat will show a
"MOTHER is not ready" error naming the missing file.

## Why not Git LFS

Render does fetch LFS files during its git clone, so LFS would work —
but GitHub's free LFS quota is 1 GB storage / 1 GB bandwidth per
month, and every Render deploy re-downloads the file, so ~8 deploys
a month would exhaust it and break deploys. Direct chunks have no
quota risk. (See the main README for the fuller reasoning.)

## Swapping in a newer champion later (e.g. S12)

1. Export the new weights as a weights-only `.npz` (drop the tied
   `output.weight`; the loader re-ties it from the token embedding)
   and split it the same way: `split -b <bytes> -d -a 2 <new>.npz
   models/<new>.npz.part` (adjust the part count in the build command
   if the file needs more or fewer parts).
2. Write the new SHA-256 file (`<new>.npz.sha256`) in the same format:
   `<sha256>  models/<new>.npz`. The `models/` prefix matters: the
   build runs `sha256sum -c` from the repo root.
3. Update `MOTHER_MODEL_PATH` in `render.yaml` (and in the Render
   dashboard under Environment) to the new filename, and update the
   build command's `cat` list + `.sha256` filename. Update the
   `STAGE` / `EPOCH` / `SELECTION_VAL_LOSS` constants in
   `mother_model.py` to the new round's values.
4. Remove the old chunk files from the repo (the old champion stays
   in git history; that is acceptable while champions are ~118 MB).
5. Commit and push. Render redeploys automatically.
