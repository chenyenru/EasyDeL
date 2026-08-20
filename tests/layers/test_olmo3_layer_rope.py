# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OLMo 3 routes RoPE by attention type: YaRN for full, default for sliding.

Reference: transformers 4.57 `Olmo3Model.rotary_embs`, which builds
`Olmo3RotaryEmbedding(config, rope_type="default")` for `sliding_attention` and
the config-scaled one for `full_attention`.
"""

import numpy as np

from easydel.modules.olmo3.olmo3_configuration import Olmo3Config

YARN_SCALING = {
    "rope_type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 64,
    "attention_factor": 1.2,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
}


def _config(rope_scaling: dict | None = None) -> Olmo3Config:
    config = Olmo3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=500_000.0,
        sliding_window=8,
    )
    if rope_scaling is not None:
        # Assigned after construction: `_rope_scaling_validation` still accepts only
        # "linear" and "dynamic", while the released OLMo 3 checkpoints declare
        # "yarn". That validator is a separate fix; this test covers the routing.
        config.rope_scaling = dict(rope_scaling)
    return config


def _cache(module_cache) -> np.ndarray:
    value = module_cache
    for attribute in ("get_value", "value"):
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            value = candidate() if callable(candidate) else candidate
            break
    return np.asarray(value)


def test_sliding_layers_get_unscaled_frequencies():
    """The two caches must differ, or the scaling leaked into the sliding layers."""
    config = _config(YARN_SCALING)
    scaled = _cache(config.get_basic_frequencies())
    unscaled = _cache(config.get_unscaled_frequencies())
    assert scaled.shape == unscaled.shape
    assert not np.allclose(scaled, unscaled)


def test_unscaled_frequencies_ignore_rope_scaling():
    """Dropping `rope_scaling` from the config must reproduce the unscaled cache."""
    unscaled = _cache(_config(YARN_SCALING).get_unscaled_frequencies())
    plain = _cache(_config().get_basic_frequencies())
    assert np.array_equal(unscaled, plain)


def test_layer_types_select_the_frequency_cache():
    """Layer 3 is full attention in the default 3-sliding/1-full pattern."""
    config = _config(YARN_SCALING)
    assert config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
