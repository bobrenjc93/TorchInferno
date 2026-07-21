from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from torchinferno.runtime.disaggregated import _pack_live_cache, _unpack_live_cache


def test_disaggregated_cache_transfer_copies_only_live_kv_region() -> None:
    source_layers = []
    target_layers = []
    for layer_id in range(3):
        keys = torch.full((2, 2, 7, 4), -100.0)
        values = torch.full_like(keys, -200.0)
        keys[:, :, :3, :] = layer_id + 1
        values[:, :, :3, :] = layer_id + 11
        source_layers.append(SimpleNamespace(keys=keys, values=values))
        target_layers.append(
            SimpleNamespace(
                keys=torch.full_like(keys, 901.0),
                values=torch.full_like(values, 902.0),
            )
        )

    source = SimpleNamespace(layers=source_layers)
    target = SimpleNamespace(layers=target_layers)
    live_elements = 3 * 2 * 2 * 2 * 3 * 4
    buffer = torch.empty(live_elements)

    _pack_live_cache(source, buffer, batch_size=2, tokens=3)
    _unpack_live_cache(target, buffer, batch_size=2, tokens=3)

    for source_layer, target_layer in zip(source.layers, target.layers):
        torch.testing.assert_close(target_layer.keys[:, :, :3, :], source_layer.keys[:, :, :3, :])
        torch.testing.assert_close(target_layer.values[:, :, :3, :], source_layer.values[:, :, :3, :])
        assert torch.all(target_layer.keys[:, :, 3:, :] == 901.0)
        assert torch.all(target_layer.values[:, :, 3:, :] == 902.0)


def test_disaggregated_cache_transfer_rejects_contract_size_mismatch() -> None:
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=torch.zeros(1, 1, 4, 2),
                values=torch.zeros(1, 1, 4, 2),
            )
        ]
    )

    with pytest.raises(ValueError, match="transfer contract"):
        _pack_live_cache(cache, torch.empty(3), batch_size=1, tokens=2)


def test_disaggregated_cache_transfer_rejects_capacity_overflow() -> None:
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=torch.zeros(1, 1, 2, 2),
                values=torch.zeros(1, 1, 2, 2),
            )
        ]
    )

    with pytest.raises(ValueError, match="exceeds cache storage"):
        _pack_live_cache(cache, torch.empty(16), batch_size=1, tokens=4)
