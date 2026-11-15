"""MOTHER V008 model definition + generation, pure numpy (no torch).

This is the V008 engine (29,578,240 parameters: 8 layers, 512 wide,
8 heads, FFN 2048, 8,000-word vocabulary, tied embeddings), ported
from the verified local numpy engine (mother_numpy_model.py) that
reproduced S11's recorded S12 generations byte-for-byte.

The weights file is an .npz of the model_state_dict only (the tied
output.weight is dropped from the file and re-tied from the token
embedding on load, exactly as the .pt path did). The loader checks
the parameter count (29,578,240) on startup and refuses to run a
file with the wrong shape.

The generate() function replicates the verified generation path
used by every stage's disk-verification cell exactly:
  prompt = f"Lesson: {question}\\nExplanation:"
  ids = [BOS] + sp.encode(prompt, out_type=int)
  greedy argmax, max 48 new tokens, stop at EOS
  answer = sp.decode(ids) split on "Explanation:", last part, stripped

The cached greedy path (greedy_cached) is byte-identical to the
uncached one and roughly 3x faster; generate() uses it.

Verified 2026-11-15: this engine, loaded from the .npz, reproduces
S11's recorded S12 answers exactly (3/3: 312+4, 7+5, after 549),
including the known failure 7+5 -> 13, which is the proof it is the
real model answering.
"""

import numpy as np

# V008 configuration (from the MOTHER scale-up plan, 2026-11-14).
VOCAB = 8000
WIDTH = 512
LAYERS = 8
HEADS = 8
FFN = 2048
CONTEXT = 512

# Special token ids (from the tokenizer: BOS=2, EOS=3, PAD=0).
BOS = 2
EOS = 3
PAD = 0

EXPECTED_PARAMS = 29578240

# The .npz holds weights only (no training metadata), so the facts
# about the weights it contains are recorded here. They describe the
# S11 checkpoint the .npz was exported from (SHA-256
# 48f539fe1a95a0ad91d8edf3d501721dcfaef0f159c6e5816505edcbf2f177ed,
# verified at build time by sha256sum -c).
STAGE = "MOTHER_V008_S11"
EPOCH = 3
SELECTION_VAL_LOSS = 0.20838302557205088

NEG_INF = np.float32(-np.inf)


def _layernorm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _relu(x):
    # PyTorch nn.TransformerEncoderLayer's DEFAULT activation is F.relu
    # (the training cell passes no activation argument). An earlier
    # version of this engine used GELU and diverged after a few tokens.
    return np.maximum(x, 0)


def _softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


class MotherV008Numpy:
    def __init__(self, npz_path):
        with np.load(npz_path) as z:
            msd = {k: z[k] for k in z.files}
        # Re-tie: the .npz omits output.weight (it duplicates the token
        # embedding); the model reads the output projection from the
        # token embedding, exactly as the .pt path does.
        msd["output.weight"] = msd["token_embedding.weight"]
        n_params = sum(v.size for k, v in msd.items()
                       if k != "output.weight")
        if n_params != EXPECTED_PARAMS:
            raise ValueError(
                f"Unexpected parameter count {n_params} "
                f"(expected {EXPECTED_PARAMS}); wrong architecture?")
        self.w = msd
        self.stage = STAGE
        self.epoch = EPOCH
        self.sel_val = SELECTION_VAL_LOSS

    def last_logits(self, ids):
        """Logits at the last position for a token-id list (1-D int array).

        Full-sequence forward (no cache). Kept as the reference
        implementation: the cached greedy path must match it exactly.
        """
        logits, _ = self._forward_full(ids)
        return logits

    def _forward_full(self, ids):
        """Full forward. Returns (last-position logits, per-layer K/V cache).

        Cache entries are (K, V) with shape (HEADS, T, head_dim).
        """
        w = self.w
        ids = np.asarray(ids, dtype=np.int64)
        t = ids.shape[0]
        h = w["token_embedding.weight"][ids] \
            + w["position_embedding.weight"][:t]
        # Causal mask (j > i) OR key is PAD.
        pad_key = ids == PAD  # (T,) bool, key positions to ignore
        causal = np.triu(np.ones((t, t), dtype=bool), k=1)
        mask = causal | pad_key[None, :]  # (T, T) True = do not attend
        head_dim = WIDTH // HEADS  # 64
        scale = np.float32(1.0 / np.sqrt(head_dim))
        cache = []
        for li in range(LAYERS):
            p = f"blocks.layers.{li}."
            qkv = h @ w[p + "self_attn.in_proj_weight"].T \
                + w[p + "self_attn.in_proj_bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(t, HEADS, head_dim).transpose(1, 0, 2)
            k = k.reshape(t, HEADS, head_dim).transpose(1, 0, 2)
            v = v.reshape(t, HEADS, head_dim).transpose(1, 0, 2)
            cache.append((k, v))
            scores = (q @ k.transpose(0, 2, 1)) * scale
            scores = np.where(mask[None, :, :], NEG_INF, scores)
            attn = _softmax(scores) @ v  # (H, T, 64)
            attn = attn.transpose(1, 0, 2).reshape(t, WIDTH)
            out = attn @ w[p + "self_attn.out_proj.weight"].T \
                + w[p + "self_attn.out_proj.bias"]
            h = _layernorm(h + out, w[p + "norm1.weight"],
                           w[p + "norm1.bias"])
            ffn = _relu(h @ w[p + "linear1.weight"].T
                        + w[p + "linear1.bias"])
            ffn = ffn @ w[p + "linear2.weight"].T \
                + w[p + "linear2.bias"]
            h = _layernorm(h + ffn, w[p + "norm2.weight"],
                           w[p + "norm2.bias"])
        last = _layernorm(h[-1], w["norm.weight"], w["norm.bias"])
        return last @ w["token_embedding.weight"].T, cache  # (8000,)

    def _forward_step(self, token_id, position, cache, pad_flags):
        """Forward for ONE new token, reusing the per-layer K/V cache.

        pad_flags: list of bools, one per key position so far (True where
        the key token is PAD). The new token's flag is appended here.
        Returns the new token's logits; cache is updated in place.
        """
        w = self.w
        h = w["token_embedding.weight"][token_id] \
            + w["position_embedding.weight"][position]
        pad_flags = pad_flags + [token_id == PAD]
        key_pad = np.asarray(pad_flags, dtype=bool)  # (T+1,)
        head_dim = WIDTH // HEADS
        scale = np.float32(1.0 / np.sqrt(head_dim))
        for li in range(LAYERS):
            p = f"blocks.layers.{li}."
            qkv = h @ w[p + "self_attn.in_proj_weight"].T \
                + w[p + "self_attn.in_proj_bias"]
            q, k_new, v_new = np.split(qkv, 3)
            q = q.reshape(HEADS, head_dim)
            k_new = k_new.reshape(HEADS, head_dim)
            v_new = v_new.reshape(HEADS, head_dim)
            k_old, v_old = cache[li]
            k = np.concatenate([k_old, k_new[:, None, :]], axis=1)
            v = np.concatenate([v_old, v_new[:, None, :]], axis=1)
            cache[li] = (k, v)
            scores = (q[:, None, :] @ k.transpose(0, 2, 1)) * scale
            scores = np.where(key_pad[None, None, :], NEG_INF, scores)
            attn = _softmax(scores) @ v  # (H, 1, 64)
            attn = attn.reshape(WIDTH)
            out = attn @ w[p + "self_attn.out_proj.weight"].T \
                + w[p + "self_attn.out_proj.bias"]
            h = _layernorm(h + out, w[p + "norm1.weight"],
                           w[p + "norm1.bias"])
            ffn = _relu(h @ w[p + "linear1.weight"].T
                        + w[p + "linear1.bias"])
            ffn = ffn @ w[p + "linear2.weight"].T \
                + w[p + "linear2.bias"]
            h = _layernorm(h + ffn, w[p + "norm2.weight"],
                           w[p + "norm2.bias"])
        last = _layernorm(h, w["norm.weight"], w["norm.bias"])
        return last @ w["token_embedding.weight"].T, pad_flags

    def greedy(self, sp, lesson_text, max_new_tokens=48):
        """Exact replica of the cell's greedy_on (no cache)."""
        prompt = f"Lesson: {lesson_text}\nExplanation:"
        ids = [BOS] + sp.encode(prompt, out_type=int)
        for _ in range(max_new_tokens):
            logits = self.last_logits(ids)
            nxt = int(np.argmax(logits))
            ids.append(nxt)
            if nxt == EOS:
                break
        out = sp.decode(ids)
        return out.split("Explanation:", 1)[-1].strip()

    def greedy_cached(self, sp, lesson_text, max_new_tokens=48):
        """Same as greedy(), but with a per-layer K/V cache.

        Must produce byte-identical generations to greedy().
        """
        prompt = f"Lesson: {lesson_text}\nExplanation:"
        ids = [BOS] + sp.encode(prompt, out_type=int)
        logits, cache = self._forward_full(ids)
        pad_flags = [t == PAD for t in ids]
        for _ in range(max_new_tokens):
            nxt = int(np.argmax(logits))
            ids.append(nxt)
            if nxt == EOS:
                break
            logits, pad_flags = self._forward_step(
                nxt, len(ids) - 1, cache, pad_flags)
        out = sp.decode(ids)
        return out.split("Explanation:", 1)[-1].strip()


def load_champion(weights_path):
    """Load the V008 S11 .npz weights and return (model, info).

    The weights live in an .npz of the model_state_dict only; the
    tied output.weight is re-tied from the token embedding on load.
    Raises if the parameter count is not the expected 29,578,240.
    """
    model = MotherV008Numpy(weights_path)
    n_params = sum(v.size for k, v in model.w.items()
                   if k != "output.weight")
    info = {
        "stage": model.stage,
        "epoch": model.epoch,
        "selection_val_loss": model.sel_val,
        "params": n_params,
    }
    return model, info


def generate(model, sp, question, max_new_tokens=48):
    """Greedy generation, exactly matching the verify-cell path.

    Uses the cached path, which is byte-identical to the uncached one.
    """
    return model.greedy_cached(sp, question,
                               max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    import sys
    import time

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=sys.argv[2])
    model, info = load_champion(sys.argv[1])
    print(f"Loaded {info['stage']}: epoch={info['epoch']} "
          f"sel_val={info['selection_val_loss']} "
          f"params={info['params']:,}")
    t0 = time.time()
    ans = generate(model, sp,
                   sys.argv[3] if len(sys.argv) > 3 else "What is 2 + 2?")
    print(f"answer: {ans!r}  ({time.time() - t0:.1f}s)")
