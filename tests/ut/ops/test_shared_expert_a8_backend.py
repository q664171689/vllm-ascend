# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SituAndMul

from vllm_ascend.ops.fused_moe import shared_experts as shared_experts_module
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.shared_experts import (
    AscendSharedExperts,
    SharedExpertMLPPath,
    _create_shared_expert_a8_backend,
)
from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendKimiK3W4A8DynamicLinearMethod
from vllm_ascend.quantization.quant_type import QuantType


def _make_layer():
    scheme = AscendKimiK3W4A8DynamicLinearMethod.__new__(AscendKimiK3W4A8DynamicLinearMethod)
    projections = []
    for _ in range(2):
        projection = MagicMock()
        projection.quant_method = SimpleNamespace(quant_method=scheme)
        projection.weight_scale = torch.ones(4)
        projections.append(projection)
    with set_current_vllm_config(VllmConfig()):
        activation = SituAndMul(beta=4.0, linear_beta=25.0)
    return SimpleNamespace(gate_up_proj=projections[0], down_proj=projections[1], act_fn=activation)


def _make_executor(layer):
    executor = AscendSharedExperts.__new__(AscendSharedExperts)
    executor.layer = layer
    executor.quant_type = QuantType.W4A8
    executor.lora_context = None
    executor._has_a8_backend_provider, executor._a8_backend = _create_shared_expert_a8_backend(layer)
    return executor


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_mixed_projection_pair_falls_back_to_wrappers(projection):
    layer = _make_layer()
    getattr(layer, projection).quant_method = SimpleNamespace(quant_method=object())
    executor = _make_executor(layer)

    assert executor._has_a8_backend_provider
    assert executor._a8_backend is None
    assert executor._select_mlp_path() is SharedExpertMLPPath.LINEAR_WRAPPER


@pytest.mark.parametrize("unsupported", ["activation", "expert_gate", "gate_up_scale", "down_scale"])
def test_unsupported_combinations_do_not_use_builtin_a8_path(unsupported):
    layer = _make_layer()
    if unsupported == "activation":
        layer.act_fn = torch.nn.SiLU()
    elif unsupported == "expert_gate":
        layer.expert_gate = torch.nn.Identity()
    elif unsupported == "gate_up_scale":
        del layer.gate_up_proj.weight_scale
    else:
        del layer.down_proj.weight_scale
    executor = _make_executor(layer)

    assert executor._has_a8_backend_provider
    assert executor._a8_backend is None
    assert executor._select_mlp_path() is SharedExpertMLPPath.LINEAR_WRAPPER


def test_no_provider_preserves_builtin_path():
    layer = _make_layer()
    layer.gate_up_proj.quant_method = None
    layer.down_proj.quant_method = None
    executor = _make_executor(layer)

    assert not executor._has_a8_backend_provider
    assert executor._a8_backend is None
    assert executor._select_mlp_path() is SharedExpertMLPPath.A8_INT_FUSED


def test_active_lora_bypasses_backend_without_recreating_it(monkeypatch):
    executor = _make_executor(_make_layer())
    backend = executor._a8_backend
    monkeypatch.setattr(shared_experts_module, "has_lora", lambda context: context is not None)

    assert executor._select_mlp_path() is SharedExpertMLPPath.A8_BACKEND
    executor.set_lora_context(object())
    assert executor._select_mlp_path() is SharedExpertMLPPath.LINEAR_WRAPPER
    executor.set_lora_context(None)
    assert executor._select_mlp_path() is SharedExpertMLPPath.A8_BACKEND
    assert executor._a8_backend is backend


def test_backend_is_resolved_once_during_initialization(monkeypatch):
    layer = _make_layer()
    backend = object()
    factory = MagicMock(return_value=(True, backend))
    monkeypatch.setattr(shared_experts_module, "_create_shared_expert_a8_backend", factory)
    monkeypatch.setattr(
        shared_experts_module,
        "get_ascend_config",
        lambda: SimpleNamespace(multistream_overlap_shared_expert=False, enable_shared_expert_dp=False),
    )
    config = SimpleNamespace(
        hidden_dim=4,
        in_dtype=torch.bfloat16,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        is_sequence_parallel=False,
    )
    executor = AscendSharedExperts(layer, config, QuantType.W4A8, MagicMock())

    for _ in range(3):
        assert executor._select_mlp_path() is SharedExpertMLPPath.A8_BACKEND
    factory.assert_called_once_with(layer)
    assert executor._a8_backend is backend


def test_staged_backend_preserves_quantization_and_wait_order(monkeypatch):
    layer = _make_layer()
    executor = _make_executor(layer)
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    original_input = hidden_states.clone()
    qx, x_scale = torch.ones(2, 4, dtype=torch.int8), torch.ones(2)
    gate_up = torch.ones(2, 8, dtype=torch.bfloat16)
    qact, act_scale = torch.ones(2, 4, dtype=torch.int8), torch.ones(2)
    output = torch.randn(2, 4, dtype=torch.bfloat16)
    milestones = RoutedMoEMilestones(
        router_output_ready=object(), routed_gmm2_start=object(), routed_combine_start=object()
    )
    calls = MagicMock()
    dynamic_quant = MagicMock(return_value=(qx, x_scale))
    activation_quant = MagicMock(return_value=(qact, act_scale))
    layer.gate_up_proj.return_value = (gate_up, None)
    layer.down_proj.return_value = (output, None)
    executor._wait_for_milestone = MagicMock()
    executor._wait_for_routed_stage = MagicMock()
    for name, mock in (
        ("quantize", dynamic_quant),
        ("wait_router", executor._wait_for_milestone),
        ("gate_up", layer.gate_up_proj),
        ("wait_stage", executor._wait_for_routed_stage),
        ("activation_quant", activation_quant),
        ("down", layer.down_proj),
    ):
        calls.attach_mock(mock, name)
    monkeypatch.setattr(shared_experts_module.torch_npu, "npu_dynamic_quant", dynamic_quant)
    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace(dequant_situ_quant=activation_quant))

    assert executor._run_shared_mlp(hidden_states, milestones) is output

    assert [call[0] for call in calls.mock_calls] == [
        "quantize",
        "wait_router",
        "gate_up",
        "wait_stage",
        "activation_quant",
        "wait_stage",
        "down",
    ]
    assert dynamic_quant.call_args.args[0] is hidden_states
    executor._wait_for_milestone.assert_called_once_with(milestones.router_output_ready, "router_output_ready")
    waits = executor._wait_for_routed_stage.call_args_list
    assert waits[0].args == (milestones, milestones.routed_gmm2_start, "routed_gmm2_start")
    assert waits[1].args == (milestones, milestones.routed_combine_start, "routed_combine_start")
    assert layer.gate_up_proj.call_args.args[0][0] is qx
    assert layer.gate_up_proj.call_args.args[0][1] is x_scale
    assert layer.down_proj.call_args.args[0][0] is qact
    assert layer.down_proj.call_args.args[0][1] is act_scale
    kwargs = activation_quant.call_args.kwargs
    assert kwargs.pop("x") is gate_up
    assert kwargs == dict(
        weight_scale=None,
        activation_scale=None,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=None,
        beta=4.0,
        linear_beta=25.0,
        activate_left=True,
        quant_mode="dynamic",
    )
    torch.testing.assert_close(hidden_states, original_input)


def test_backend_reads_projections_after_loading_instead_of_caching_weights():
    layer = _make_layer()
    executor = _make_executor(layer)
    backend = executor._a8_backend
    assert backend is not None
    old_projection = layer.gate_up_proj
    expected = torch.ones(2, 4, dtype=torch.bfloat16)
    layer.gate_up_proj = MagicMock(return_value=(expected, None))

    assert backend.gate_up(torch.ones(2, 4, dtype=torch.int8), torch.ones(2)) is expected
    old_projection.assert_not_called()
