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

WEIGHT NAMING (verified against qwen_tts 0.1.x source, see core/models/modeling_qwen3_tts.py):
The talker module is `Qwen3TTSTalkerForConditionalGeneration` at attribute
`top_model.talker`. Its state_dict, when extracted from the talker module
directly (not the parent), looks like:

  model.codec_embedding.weight          [3072, 1024]   ← codec input embed
  model.text_embedding.weight           [151936, 2048] ← NOT used by kernel
  model.norm.weight                     [1024]         ← final RMSNorm
  model.layers.{i}.input_layernorm.weight                    [1024]
  model.layers.{i}.self_attn.{q,k,v,o}_proj.weight           [1024,1024 or 1024,2048]
  model.layers.{i}.self_attn.{q,k}_norm.weight               [128]
  model.layers.{i}.post_attention_layernorm.weight           [1024]
  model.layers.{i}.mlp.{gate,up,down}_proj.weight            [3072,1024 or 1024,3072]
  codec_head.weight                     [3072, 1024]   ← codec output head
  lm_head.weight                        [151936, 1024] ← TEXT head, NOT used by kernel
  text_projection.linear_fc{1,2}.weight                ← NOT used by kernel (prefill only)
  code_predictor.*                                      ← subtalker, NOT used by kernel

The kernel only sees codec-side weights (embed + 28 decoder layers + final norm +
codec_head). Everything else (text embed, text projection, code predictor) is
kept HF-side for prefill and per-step subtalker work.

INTEGRATION STRATEGY:
  * Text prefill (one-time, ~50 text tokens):  run via HuggingFace forward,
    which warms the talker's KV cache and produces the first codec hidden state.
  * Codec autoregressive decode (the hot loop, 12.5 tokens / sec of audio):
    drive each step with `torch.ops.qwen_megakernel_C.decode(...)`.

This loader extracts the codec-side weights into the pointer-packed layout the
megakernel struct expects.  See LDGLayerWeights in csrc/torch_bindings.cpp.
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
            talker_module: the `Qwen3TTSTalkerForConditionalGeneration` instance,
                accessible as `Qwen3TTSForConditionalGeneration.talker`.
                Must already be on CUDA in bfloat16. Expected layout:
                  - model.codec_embedding.weight   [3072, 1024]
                  - model.norm.weight              [1024]
                  - model.layers.{i}.{q_proj,...}
                  - codec_head.weight              [3072, 1024]
        """
        self.device = torch.device("cuda")
        self._packed_buf: bytes | None = None
        self._layer_tensor_refs: list[torch.Tensor] = []  # keep alive
        self._extract(talker_module)

    def _extract(self, talker: torch.nn.Module) -> None:
        sd = dict(talker.state_dict())

        # ---- codec input embedding -------------------------------------------
        # The talker module's codec embed lives at `model.codec_embedding.weight`.
        # We also accept a few legacy/alt names in case qwen_tts renames it.
        embed_candidates = [
            "model.codec_embedding.weight",
            "codec_embedding.weight",
            "model.codec_embed_tokens.weight",  # speculative legacy name
        ]
        embed_key = next((k for k in embed_candidates if k in sd), None)
        if embed_key is None:
            raise KeyError(
                f"Cannot find codec embed weight on the talker module. "
                f"Tried {embed_candidates}. "
                f"Available top-level keys (first 20): {list(sd)[:20]}"
            )
        embed = sd[embed_key]
        if embed.shape[0] != 3072:
            raise ValueError(
                f"Expected codec embed of shape [3072, 1024]; got {tuple(embed.shape)}. "
                "Pass `Qwen3TTSForConditionalGeneration.talker` — not the parent model, "
                "and not the talker's `.model` submodule."
            )
        self.embed_weight = embed.to(torch.bfloat16).to(self.device).contiguous()

        # ---- final norm -------------------------------------------------------
        norm_key = next(
            (k for k in ("model.norm.weight", "norm.weight") if k in sd),
            None,
        )
        if norm_key is None:
            raise KeyError(
                f"Cannot find talker final-norm weight. "
                f"Available keys containing 'norm': "
                f"{[k for k in sd if 'norm' in k.lower()][:10]}"
            )
        self.final_norm_weight = sd[norm_key].to(torch.bfloat16).to(self.device).contiguous()

        # ---- codec output head (NOT lm_head — that's the text head) ----------
        # The talker has two output heads:
        #   - codec_head : Linear(1024, 3072)    ← what we want
        #   - lm_head    : Linear(1024, 151936)  ← text head, untrained/ignored at decode time
        codec_head_candidates = ["codec_head.weight", "model.codec_head.weight"]
        codec_head_key = next((k for k in codec_head_candidates if k in sd), None)
        if codec_head_key is None:
            raise KeyError(
                f"Cannot find talker codec_head weight. "
                f"Tried {codec_head_candidates}. "
                "Note: 'lm_head' is the TEXT head (vocab 151936), not what the megakernel needs."
            )
        codec_head = sd[codec_head_key]
        if codec_head.shape[0] != 3072 or codec_head.shape[1] != 1024:
            raise ValueError(
                f"codec_head shape {tuple(codec_head.shape)} != expected [3072, 1024]"
            )
        self.lm_head_weight = codec_head.to(torch.bfloat16).to(self.device).contiguous()

        # ---- per-layer weights, packed into LDGLayerWeights[NUM_LAYERS] -----
        layers_prefix = self._detect_layers_prefix(sd)
        self.layer_weights_packed = self._pack_layer_weights(sd, layers_prefix)

    def _detect_layers_prefix(self, sd: dict) -> str:
        # The talker module's transformer layers are at `model.layers.{i}`.
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
