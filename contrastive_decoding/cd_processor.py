"""
cd_processor.py
---------------
The Audio-Aware Decoding logits processor, ported from the AdaptivePerturbation
project's adaptive_perturbation/run_qwen.py and run_af3.py.

Mechanism
---------
At every generation step, HuggingFace's generate() loop hands the processor the
next-token logits from the CLEAN forward pass (real audio). The processor runs a
second forward pass over a NEGATIVE branch -- the same prompt with a degraded
waveform -- and combines them:

    modified_logits = (1 + alpha) * clean_logits - alpha * negative_logits

Tokens the model would predict anyway without real acoustic evidence are pushed
down; tokens whose probability depends on the audio are pushed up. This is the
AAD formula from arXiv 2506.07233, matching the reference implementation at
github.com/GillbertHsu/Audio-Aware-Decoding (which uses alpha=0.5 by default).

Branch synchronization
----------------------
The negative branch is seeded with the full negative prompt as INPUT EMBEDDINGS,
taken from hidden_states[0] of a forward pass over the perturbed audio -- that is
the LM's input embedding sequence after the audio encoder and projector have run
and the audio frames have been merged with the text tokens. Working in embedding
space is what makes this possible at all: the audio has already been consumed
into embeddings, so the negative branch can be extended token-by-token like a
plain text sequence.

At each subsequent step the token the CLEAN branch just emitted is embedded and
appended to the negative sequence, so both branches stay conditioned on the same
partial answer y_{<t}. Without this the negative branch would drift onto its own
hypothesis and the subtraction would be meaningless.

Cost
----
The negative branch keeps no KV cache: every step re-forwards the whole negative
sequence. For an N-token answer that is N full forward passes on top of the
clean generation. This matches the AdaptivePerturbation implementation exactly
and is deliberately left alone -- caching here would be an unverified change to
a numerically sensitive path. It is the dominant cost of these runs.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor


class ContrastiveAudioLogitsProcessor(LogitsProcessor):
    """Applies (1 + alpha) * clean - alpha * negative at every decoding step.

    model:       the UNWRAPPED model (accelerator.unwrap_model(...)), used for
                 the negative forward pass and its input embedding table.
    embeds_neg:  [batch, seq, hidden] negative-branch input embeddings,
                 i.e. hidden_states[0] of a forward pass over perturbed audio.
    atts_neg:    [batch, seq] attention mask matching embeds_neg.
    alpha:       contrastive strength.
    eos_token_id: used to mask padding once a sequence has finished.
    """

    def __init__(self, model, embeds_neg, atts_neg, alpha, eos_token_id=None):
        self.model = model
        self.alpha = float(alpha)
        self.embeds_neg = embeds_neg
        self.atts_neg = atts_neg
        self.eos_token_id = eos_token_id
        self.first_call = True

        if embeds_neg.shape[1] != atts_neg.shape[1]:
            raise ValueError(
                "Negative branch shape mismatch: embeds "
                f"{tuple(embeds_neg.shape)} vs attention mask {tuple(atts_neg.shape)}"
            )

    def __call__(self, input_ids, scores):
        # `scores` = raw next-token logits from the CLEAN audio forward pass.
        target_device = scores.device

        with torch.no_grad():
            if self.first_call:
                # Step 0: the negative embeddings are the full prompt already.
                self.embeds_neg = self.embeds_neg.to(target_device)
                self.atts_neg = self.atts_neg.to(target_device)
            else:
                # Append the token the clean branch just emitted, so both
                # branches condition on the same prefix.
                new_tokens = input_ids[:, -1:].to(target_device)
                new_embeds = self.model.get_input_embeddings()(new_tokens)
                self.embeds_neg = torch.cat(
                    [self.embeds_neg, new_embeds.to(self.embeds_neg.dtype)], dim=1
                )
                if self.eos_token_id is None:
                    new_atts = torch.ones_like(new_tokens, dtype=self.atts_neg.dtype)
                else:
                    new_atts = (new_tokens != self.eos_token_id).to(self.atts_neg.dtype)
                self.atts_neg = torch.cat([self.atts_neg, new_atts], dim=1)

            out_neg = self.model(
                inputs_embeds=self.embeds_neg,
                attention_mask=self.atts_neg,
            )
            logits_neg = out_neg.logits[:, -1, :]

        self.first_call = False

        # AAD contrastive formula. alpha=1.0 -> modified = 2*clean - negative.
        return (1 + self.alpha) * scores - self.alpha * logits_neg.to(scores.dtype)
