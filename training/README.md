# Feedback becomes training

This folder is where what you teach MOTHER on the chat page turns into
lessons she actually trains on. Talking to her does not train her —
chatting never changes her weights. But every thumbs up, thumbs down,
and "why" you leave is saved, and this folder is where those ratings
become training data for her next round.

## The pipeline, in plain words

1. **You chat and rate.** On the chat page, every answer gets a thumbs
   up / thumbs down. Thumbs up is one tap, done — a right answer
   needs no explanation. Thumbs down opens a "why" field asking what
   she should have said: that correction is what teaches her. The app
   stores each question, her answer, your rating, and your reason
   (when given) in its database.
2. **Muse exports the feedback.** From the running app, `/feedback/export`
   downloads everything as JSON (or CSV). Export before any redeploy:
   on Render's free tier the database is wiped on restart.
3. **Muse converts it into lessons.** Each rated answer becomes one
   line in `feedback-lessons.jsonl`, in MOTHER's lesson format
   (`lesson` = the question, `explanation` = the answer to learn):
   - **Thumbs up** — her answer was right, so her answer becomes a
     positive lesson: she trains on getting it right again.
   - **Thumbs down + why** — her answer was wrong, so your reason
     becomes the correction: the lesson teaches the right answer,
     with your reason recorded as what was wrong with hers.
   - Answers with no rating are not converted. A rating with no
     reason still converts (thumbs down with no reason becomes a
     lesson teaching the right answer, marked as having no reason).
4. **The lessons are appended here.** One JSON object per line, newest
   at the end. Nothing is ever deleted from this file; the next
   training round reads the whole thing.
5. **The next Kaggle training round trains on it.** The lessons are
   mixed into that round's training data like any other lesson file:
   audited, deduplicated, and checked against the validation sets
   before training.

## File format

`feedback-lessons.jsonl` — one JSON object per line, no comments:

```json
{"lesson": "What is 7 + 5?", "explanation": "Count on from 7: 8, 9, 10, 11, 12. So 7 + 5 = 12.", "source": "chat-feedback", "rating": "up", "reason": "", "exchange_id": 12, "created_at": "2026-11-14T07:00:00+00:00"}
```

Fields:

| Field | What it is |
| --- | --- |
| `lesson` | The question, exactly as asked. |
| `explanation` | The answer to learn. For thumbs up, her own answer. For thumbs down, the corrected answer. |
| `source` | Where the lesson came from: `chat-feedback`, or `example` for the format example. |
| `rating` | `up` or `down` — the rating that produced this lesson. |
| `reason` | Your "why", exactly as written (empty when none was given). |
| `exchange_id` | The chat exchange number, so a lesson can be traced back to the conversation. |
| `created_at` | When the exchange happened. |
| `example` | Only on the format example line (`true`). Training skips example lines. |

## Rules (so the feedback teaches her true things)

- **Never train on test questions.** If a chat question matches an
  item in any verification battery, it must NOT become a lesson —
  training on the test would make her scores meaningless. The audit
  step checks this before training.
- **A correction must be right before it becomes a lesson.** Your
  "why" is the teacher: if the reason itself is wrong, the lesson
  teaches her something wrong. Muse checks each thumbs-down
  correction for correctness before converting it.
- **Example lines are skipped.** Any line with `"example": true`
  is a format example only and is never trained on.
- **One question, one lesson.** If the same question is rated more
  than once, the newest rating wins; the audit deduplicates.

## Why this is not automatic

The app running on Render has no GitHub credentials, so it cannot
push lessons to this repo itself — and it should not. The
export-then-commit step is done by Muse: export the feedback, check
the corrections, convert, and commit. That keeps a human (Muse, with
Johnathan's direction) in the loop before anything she is taught
becomes permanent.
