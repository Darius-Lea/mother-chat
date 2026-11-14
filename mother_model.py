"""MOTHER model definition + generation.

The MotherV0061 class is copied verbatim from the MOTHER V007
foundational-training notebook (the Load V006.1 cell). Every MOTHER
champion since V006.1 shares this architecture; only the weights
change, so this file never needs to change when a new champion is
swapped in.

The generate() function replicates the verified generation path
used by every stage's disk-verification cell exactly:
  prompt = f"Lesson: {question}\\nExplanation:"
  ids = [BOS] + sp.encode(prompt, out_type=int)
  greedy argmax, max 48 new tokens, stop at EOS
  answer = sp.decode(ids) split on "Explanation:", last part, stripped
"""

import copy

import torch
import torch.nn as nn

# Original V006.1 configuration (from the notebook).
VOCAB = 8000
WIDTH = 256
LAYERS = 4
HEADS = 4
FFN = 1024
CONTEXT = 512

# Special token ids (from the tokenizer: BOS=2, EOS=3, PAD=0).
BOS = 2
EOS = 3
PAD = 0

EXPECTED_PARAMS = 5338624


class MotherV0061(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(VOCAB, WIDTH)
        self.position_embedding = nn.Embedding(CONTEXT, WIDTH)

        layer = nn.TransformerEncoderLayer(
            d_model=WIDTH,
            nhead=HEADS,
            dim_feedforward=FFN,
            batch_first=True
        )

        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=LAYERS
        )

        self.norm = nn.LayerNorm(WIDTH)
        self.output = nn.Linear(WIDTH, VOCAB, bias=False)

        self.output.weight = self.token_embedding.weight

    def forward(self, x, pad_mask):
        positions = torch.arange(x.size(1), device=x.device)

        h = (
            self.token_embedding(x)
            + self.position_embedding(positions)
        )

        causal_mask = torch.triu(
            torch.ones(
                x.size(1), x.size(1),
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        h = self.blocks(
            h,
            mask=causal_mask,
            src_key_padding_mask=pad_mask
        )

        return self.output(self.norm(h))


def load_champion(weights_path):
    """Load a champion .pt file onto CPU and return (model, info).

    The checkpoint is a dict saved with torch.save; the weights live
    under "model_state_dict". The base model is built fresh and the
    state dict is loaded into it (the V006.1 base is never modified).
    Raises if the parameter count is not the expected 5,338,624.
    """
    checkpoint = torch.load(weights_path, map_location="cpu",
                            weights_only=False)
    model = MotherV0061()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != EXPECTED_PARAMS:
        raise ValueError(
            f"Unexpected parameter count {n_params} "
            f"(expected {EXPECTED_PARAMS}); wrong architecture?")
    info = {
        "stage": checkpoint.get("stage"),
        "epoch": checkpoint.get("epoch"),
        "selection_val_loss": checkpoint.get("selection_val_loss"),
        "params": n_params,
    }
    return model, info


@torch.no_grad()
def generate(model, sp, question, max_new_tokens=48):
    """Greedy generation, exactly matching the verify-cell path."""
    prompt = f"Lesson: {question}\nExplanation:"
    ids = [BOS] + sp.encode(prompt, out_type=int)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long)
        logits = model(x, x.eq(PAD))
        nxt = int(logits[0, -1].argmax())
        ids.append(nxt)
        if nxt == EOS:
            break
    out = sp.decode(ids)
    return out.split("Explanation:", 1)[-1].strip()
