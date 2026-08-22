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

import math

import numpy as np
import pytest

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
        config.rope_scaling = dict(rope_scaling)
        config._rope_scaling_validation()
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


def test_released_yarn_config_is_accepted():
    """`allenai/Olmo-3-7B-Think` ships rope_type "yarn"; it must construct."""
    config = Olmo3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=65_536,
        rope_theta=500_000.0,
        sliding_window=8,
        rope_scaling={
            "attention_factor": 1.2079441541679836,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "yarn",
        },
    )
    assert config.rope_scaling["rope_type"] == "yarn"


def test_hf_attention_factor_becomes_a_residual_attn_factor():
    """EasyDeL multiplies its own mscale by `attn_factor`, so divide it out.

    The released config's attention_factor is exactly HuggingFace's default
    `0.1 * ln(factor) + 1`, which is the same mscale EasyDeL infers, so the
    residual is 1.0 — forwarding 1.2079 unchanged would square the scale.
    """
    config = _config(
        {"rope_type": "yarn", "factor": 8.0, "attention_factor": 1.2079441541679836}
    )
    assert config.rope_scaling["attn_factor"] == pytest.approx(1.0, abs=1e-9)
    # The original key survives so the config still round-trips.
    assert config.rope_scaling["attention_factor"] == 1.2079441541679836


def test_custom_attention_factor_survives_the_conversion():
    """A hand-tuned attention factor keeps its ratio to the inferred mscale."""
    inferred = 0.1 * math.log(8.0) + 1.0
    config = _config({"rope_type": "yarn", "factor": 8.0, "attention_factor": 2 * inferred})
    assert config.rope_scaling["attn_factor"] == pytest.approx(2.0)


def test_explicit_attn_factor_is_left_alone():
    """A config written for EasyDeL already carries the residual factor."""
    config = _config(
        {"rope_type": "yarn", "factor": 8.0, "attention_factor": 1.2, "attn_factor": 0.5}
    )
    assert config.rope_scaling["attn_factor"] == 0.5


def test_unknown_rope_type_is_still_rejected():
    with pytest.raises(ValueError, match="type field must be one of"):
        _config({"rope_type": "banana", "factor": 8.0})
