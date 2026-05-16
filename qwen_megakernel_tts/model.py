"""
Loads the Qwen3-TTS talker weights into the layout the megakernel expects.

ARCHITECTURE NOTE (read this first):
====================================
The Qwen3-TTS talker is NOT a drop-in for Qwen3-0.6B. From config.json:

  talker_config:
    hidden_size            : 1024     ← matches Qwen3-0.6B
    num_hidden_layers      : 28       ← matches
    num_attention_heads    : 16       ← matches
    num_key_value_heads    :  8       ← matches
    head_dim               : 128      ← matches
    intermediate_size      : 3072     ← matches
    rope_theta             : 1_000_000 ← differs (Qwen3-0.6B uses 10_000)
    rope_scaling.mrope_section: [24,20,20] ← MRoPE, kernel uses standard RoPE
    text_hidden_size       : 2048     ← separate text embedding, projected to 1024
    text_vocab_size        : 151_936  ← text input vocab
    vocab_size             :   3_072  ← codec output vocab

The transformer backbone (after the embed projection) matches Qwen3-0.6B in
every architectural dimension. The megakernel can therefore drive the talker's
*per-step decode* of codec tokens — but it CANNOT do the text prefill, because
text tokens flow through a separate 2048-dim text embedding + 2048→1024
projection that the kernel doesn't implement.

INTEGRATION STRATEGY:
  * Text prefill (one-time, ~50 text tokens):  run via HuggingFace forward,
    which warms the talker's KV cache and produces the first codec hidden state.
  * Codec autoregressive decode (the hot loop, 12.5 tokens / sec of audio):
    drive each step with `torch.ops.qwen_megakernel_C.decode(...)`.

This loader extracts the codec-side weights (codec embed + lm head + per-layer
transformer weights) into the pointer-packed layout the megakernel struct
expects.  See LDGLayerWeights in csrc/torch_bindings.cpp.
"""

import struct
import torch


NUM_LAYERS = 28
NUM_LAYER_TENSORS = 11  # see LDGLayerWeights struct in csrc/torch_bindings.cpp

# Exact key order required by the LDGLayerWeights C++ struct.
_LAYER_WEIGHT_KEYS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


class TalkerKernelWeights:
    """
    Extracts the talker's codec-side weights from an already-loaded
    Qwen3TTSForConditionalGeneration model and packs them for the megakernel.

    The model is loaded via the official qwen_tts package; we just borrow its
    weight tensors. We do NOT re-download or re-instantiate the model.
    """

    def __init__(self, talker_module: torch.nn.Module):
        """
        Args:
            talker_module: the talker sub-model from a loaded
                Qwen3TTSForConditionalGeneration instance.  Must already be
                on CUDA in bfloat16. Expected attributes:
                  - codec_embed_tokens (or model.embed_tokens) [3072, 1024]
                  - model.layers[i].{q_proj, k_proj, ...}
                  - model.norm.weight
                  - lm_head.weight
        """
        self.device = torch.device("cuda")
        self._packed_buf: bytes | None = None
        self._layer_tensor_refs: list[torch.Tensor] = []  # keep alive
        self._extract(talker_module)

    def _extract(self, talker: torch.nn.Module) -> None:
        sd = dict(talker.state_dict())

        # ---- codec input embedding -------------------------------------------
        embed_candidates = [
            "codec_embed_tokens.weight",
            "model.codec_embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ]
        embed_key = next((k for k in embed_candidates if k in sd), None)
        if embed_key is None:
            raise KeyError(
                f"Cannot find codec embed weight. Tried {embed_candidates}. "
                f"Available top-level keys: {list(sd)[:20]}"
            )
        embed = sd[embed_key]
        if embed.shape[0] != 3072:
            raise ValueError(
                f"Expected codec embed of shape [3072, 1024]; got {tuple(embed.shape)}. "
                "Check that you passed the talker module, not the full text-side embed."
            )
        self.embed_weight = embed.to(torch.bfloat16).to(self.device).contiguous()

        # ---- final norm + lm head --------------------------------------------
        norm_key = next(
            (k for k in ("model.norm.weight", "norm.weight") if k in sd),
            None,
        )
        if norm_key is None:
            raise KeyError("Cannot find talker final-norm weight.")
        self.final_norm_weight = sd[norm_key].to(torch.bfloat16).to(self.device).contiguous()

        if "lm_head.weight" in sd:
            self.lm_head_weight = sd["lm_head.weight"].to(torch.bfloat16).to(self.device).contiguous()
        else:
            # Tied embedding (codec embed reused as lm head).
            self.lm_head_weight = self.embed_weight

        # ---- per-layer weights, packed into LDGLayerWeights[NUM_LAYERS] -----
        layers_prefix = self._detect_layers_prefix(sd)
        self.layer_weights_packed = self._pack_layer_weights(sd, layers_prefix)

    def _detect_layers_prefix(self, sd: dict) -> str:
        for prefix in ("model.layers.", "layers."):
            if f"{prefix}0.{_LAYER_WEIGHT_KEYS[0]}" in sd:
                return prefix
        raise KeyError(
            "Cannot find talker transformer layers. "
            f"Sample keys: {list(sd)[:10]}"
        )

    def _pack_layer_weights(self, sd: dict, layers_prefix: str) -> torch.Tensor:
        """
        Build a flat uint8 buffer of NUM_LAYERS LDGLayerWeights structs:
          struct { const void* p0..p10; }   // 11 raw device pointers each
        Stored as uint64 little-endian.  Tensor refs are retained so the
        pointers stay valid for the kernel's lifetime.
        """
        self._layer_tensor_refs = []
        buf = bytearray(NUM_LAYERS * NUM_LAYER_TENSORS * 8)

        for layer_idx in range(NUM_LAYERS):
            for weight_idx, key in enumerate(_LAYER_WEIGHT_KEYS):
                full = f"{layers_prefix}{layer_idx}.{key}"
                if full not in sd:
                    raise KeyError(f"Missing talker layer weight: {full}")
                t = sd[full].to(torch.bfloat16).to(self.device).contiguous()
                self._layer_tensor_refs.append(t)
                offset = (layer_idx * NUM_LAYER_TENSORS + weight_idx) * 8
                struct.pack_into("<Q", buf, offset, t.data_ptr())

        return torch.frombuffer(bytes(buf), dtype=torch.uint8).to(self.device).contiguous()
