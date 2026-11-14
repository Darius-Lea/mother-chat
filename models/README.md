# models/

This folder holds the two files MOTHER needs to talk.

## What goes here

1. `stageR14_champion.pt` — the R14 champion weights (64,129,445 bytes,
   SHA-256 `dae9162bfe167051dbf98604de1ea35b4803b9afe1fb5063a700c3bbdc4a6c90`).
   This file is NOT committed directly: it is split into three chunks
   (see below) and reassembled at build time.
2. `mother_v006.model` — the SentencePiece tokenizer (368,413 bytes,
   SHA-256 `cd6ca7f8114951450fb48d6f65b2774013252cda34486ff1a63e3526ee1cee5d`),
   from the Kaggle dataset `mother-v0061-master-backup`. Committed as a
   single file.
3. `stageR14_champion.pt.sha256` — the SHA-256 of the reassembled
   weights file. The build verifies it, so a corrupted chunk fails the
   deploy loudly instead of serving a broken model.

## Why the weights are split into chunks

The 64 MB weights file is under GitHub's 100 MB per-file limit, so a
direct commit is allowed — but GitHub's web upload caps at 25 MB per
file, and the Muse GitHub connection cannot push binary files at all.
So the weights live in the repo as three chunks, each under 25 MB:

| Chunk | Bytes |
| --- | --- |
| `stageR14_champion.pt.part00` | 23,068,672 |
| `stageR14_champion.pt.part01` | 23,068,672 |
| `stageR14_champion.pt.part02` | 17,992,101 |

The chunks were made with `split -b 22m -d stageR14_champion.pt
stageR14_champion.pt.part` and verified byte-identical on reassembly
(`cat` in order reproduces the original SHA-256). The Render build
command in `render.yaml` reassembles them, checks the SHA-256, and
only then installs dependencies:

```sh
cat models/stageR14_champion.pt.part00 \
    models/stageR14_champion.pt.part01 \
    models/stageR14_champion.pt.part02 \
    > models/stageR14_champion.pt \
  && sha256sum -c models/stageR14_champion.pt.sha256 \
  && pip install -r requirements.txt
```

Upload the three chunks + the tokenizer through the GitHub web UI
(repo page → Add file → Upload files, into `models/`), or with git
CLI from a machine with GitHub access. The app will start without
them but the chat will show a "MOTHER is not ready" error naming the
missing file.

## Why not Git LFS

Render does fetch LFS files during its git clone, so LFS would work —
but GitHub's free LFS quota is 1 GB storage / 1 GB bandwidth per
month, and every Render deploy re-downloads the file, so ~15 deploys
a month would exhaust it and break deploys. Direct chunks have no
quota risk. (See the main README for the fuller reasoning.)

## Swapping in a newer champion later (e.g. R15)

1. Split the new `.pt` file the same way:
   `split -b 22m -d <new>.pt models/<new>.pt.part` (adjust the part
   count in the build command if the file needs more or fewer parts).
2. Write the new SHA-256 file (`<new>.pt.sha256`) in the same format:
   `<sha256>  models/<new>.pt`.
3. Update `MOTHER_MODEL_PATH` in `render.yaml` (and in the Render
   dashboard under Environment) to the new filename, and update the
   build command's `cat` list + `.sha256` filename.
4. Remove the old chunk files from the repo (the old champion stays
   in git history; that is acceptable while champions are 64 MB).
5. Commit and push. Render redeploys automatically.
