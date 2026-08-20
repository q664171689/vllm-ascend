#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Regression coverage for Kimi-K3 BF16 chunk-KDA tail execution."""

from dataclasses import dataclass

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

_CHUNK_SIZE = 64
_HEADS = 6
_HEAD_DIM = 128
_LOWER_BOUND = -5.0
_MAX_REASONABLE_ABS = 1.0e6
_TAIL_TEST_TOKENS = (128, 129, 130, 131, 143, 144, 145, 159, 160, 161, 191, 192, 193)
_OUTPUT_NAMES = (
    "output",
    "final_state",
    "gk",
    "aqk",
    "akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
)


@dataclass
class _ChunkKdaInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_gate: torch.Tensor
    activated_gate: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    initial_state: torch.Tensor
    cu_seqlens: tuple[int, ...]
    chunk_indices: tuple[int, ...]

    def clone(self) -> "_ChunkKdaInputs":
        return _ChunkKdaInputs(
            q=self.q.clone(),
            k=self.k.clone(),
            v=self.v.clone(),
            raw_gate=self.raw_gate.clone(),
            activated_gate=self.activated_gate.clone(),
            beta=self.beta.clone(),
            a_log=self.a_log.clone(),
            dt_bias=self.dt_bias.clone(),
            initial_state=self.initial_state.clone(),
            cu_seqlens=self.cu_seqlens,
            chunk_indices=self.chunk_indices,
        )


def _l2norm(value: torch.Tensor) -> torch.Tensor:
    dtype = value.dtype
    value_fp32 = value.float()
    return (value_fp32 * torch.rsqrt((value_fp32 * value_fp32).sum(dim=-1, keepdim=True) + 1e-6)).to(dtype)


def _build_inputs(tokens: int) -> _ChunkKdaInputs:
    torch.manual_seed(20260820 + tokens)
    torch.npu.manual_seed_all(20260820 + tokens)

    shape = (1, tokens, _HEADS, _HEAD_DIM)
    q = _l2norm(torch.randn(shape, device="npu", dtype=torch.bfloat16))
    k = _l2norm(torch.randn(shape, device="npu", dtype=torch.bfloat16))
    v = torch.randn(shape, device="npu", dtype=torch.bfloat16) * 0.2
    raw_gate = torch.randn(shape, device="npu", dtype=torch.bfloat16) * 2.0
    beta = torch.sigmoid(torch.randn((1, tokens, _HEADS), device="npu", dtype=torch.float32))
    a_log = torch.empty((_HEADS,), device="npu", dtype=torch.float32).uniform_(-0.5, 0.8)
    dt_bias = torch.empty((_HEADS * _HEAD_DIM,), device="npu", dtype=torch.float32).uniform_(-7.5, -1.5)
    initial_state = torch.zeros((1, _HEADS, _HEAD_DIM, _HEAD_DIM), device="npu", dtype=torch.float32)
    activated_gate = _LOWER_BOUND * torch.sigmoid(
        (raw_gate.float() + dt_bias.view(1, 1, _HEADS, _HEAD_DIM))
        * a_log.exp().view(1, 1, _HEADS, 1)
    )
    chunk_indices = tuple(
        item
        for chunk_index in range((tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE)
        for item in (0, chunk_index)
    )
    return _ChunkKdaInputs(
        q=q,
        k=k,
        v=v,
        raw_gate=raw_gate,
        activated_gate=activated_gate,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=(0, tokens),
        chunk_indices=chunk_indices,
    )


def _run_chunk_kda(inputs: _ChunkKdaInputs, gate_mode: str, metadata_mode: str):
    use_gate_in_kernel = gate_mode == "raw_gate"
    gate = inputs.raw_gate if use_gate_in_kernel else inputs.activated_gate
    use_varlen_metadata = metadata_mode == "varlen"
    return torch.ops._C_ascend.chunk_kda_fwd(
        inputs.q,
        inputs.k,
        inputs.v,
        gate,
        inputs.beta,
        _HEAD_DIM**-0.5,
        _CHUNK_SIZE,
        layout="BSND",
        initial_state=inputs.initial_state,
        output_final_state=True,
        cu_seqlens=inputs.cu_seqlens if use_varlen_metadata else None,
        chunk_indices=inputs.chunk_indices if use_varlen_metadata else None,
        safe_gate=True,
        lower_bound=_LOWER_BOUND,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=inputs.a_log if use_gate_in_kernel else None,
        dt_bias=inputs.dt_bias if use_gate_in_kernel else None,
        disable_recompute=True,
        return_intermediate_states=False,
        state_v_first=True,
    )


def _snapshot_outputs(outputs) -> tuple[torch.Tensor | None, ...]:
    torch.npu.synchronize()
    return tuple(output.detach().cpu().contiguous() if isinstance(output, torch.Tensor) else None for output in outputs)


def _first_nonfinite_flat(value: torch.Tensor) -> int:
    positions = (~torch.isfinite(value.float())).flatten().nonzero()
    return int(positions[0].item()) if positions.numel() else -1


def _finite_max_abs(value: torch.Tensor) -> float:
    value_fp64 = value.double()
    finite = torch.isfinite(value_fp64)
    if not finite.any().item():
        return float("nan")
    return value_fp64[finite].abs().max().item()


def _first_differing_coordinate(first: torch.Tensor, second: torch.Tensor) -> tuple[int, ...] | None:
    if first.shape != second.shape or first.dtype != second.dtype:
        return None
    different_bytes = first.view(torch.uint8).flatten() != second.view(torch.uint8).flatten()
    positions = different_bytes.nonzero()
    if not positions.numel():
        return None

    flat_index = int(positions[0].item()) // first.element_size()
    coordinate = []
    for dim in reversed(first.shape):
        coordinate.append(flat_index % dim)
        flat_index //= dim
    return tuple(reversed(coordinate))


def _differing_element_count(first: torch.Tensor, second: torch.Tensor) -> int:
    if first.shape != second.shape or first.dtype != second.dtype:
        return -1
    different_bytes = first.view(torch.uint8).flatten() != second.view(torch.uint8).flatten()
    return int(different_bytes.reshape(-1, first.element_size()).any(dim=1).sum().item())


def _describe_difference(name: str, first: torch.Tensor | None, second: torch.Tensor | None) -> str | None:
    if first is None or second is None:
        return f"{name}: missing output first={type(first).__name__} second={type(second).__name__}"
    same_metadata = first.shape == second.shape and first.dtype == second.dtype
    same_bits = same_metadata and torch.equal(first.view(torch.uint8), second.view(torch.uint8))
    first_nonfinite = int((~torch.isfinite(first.float())).sum().item())
    second_nonfinite = int((~torch.isfinite(second.float())).sum().item())
    first_max_abs = _finite_max_abs(first)
    second_max_abs = _finite_max_abs(second)
    values_are_reasonable = first_max_abs <= _MAX_REASONABLE_ABS and second_max_abs <= _MAX_REASONABLE_ABS
    if same_bits and first_nonfinite == 0 and second_nonfinite == 0 and values_are_reasonable:
        return None

    first_fp64 = first.double()
    second_fp64 = second.double()
    finite_overlap = torch.isfinite(first_fp64) & torch.isfinite(second_fp64)
    max_abs_diff = (
        (first_fp64[finite_overlap] - second_fp64[finite_overlap]).abs().max().item()
        if finite_overlap.any().item()
        else float("nan")
    )
    first_diff = _first_differing_coordinate(first, second)
    first_value = first[first_diff].item() if first_diff is not None else None
    second_value = second[first_diff].item() if first_diff is not None else None
    return (
        f"{name}: shape_first={tuple(first.shape)} shape_second={tuple(second.shape)} "
        f"dtype_first={first.dtype} dtype_second={second.dtype} "
        f"first_nonfinite={first_nonfinite} second_nonfinite={second_nonfinite} "
        f"first_bad_flat={_first_nonfinite_flat(first)} second_bad_flat={_first_nonfinite_flat(second)} "
        f"first_max_abs={first_max_abs:.8e} second_max_abs={second_max_abs:.8e} "
        f"max_abs_diff={max_abs_diff:.8e} differing_elements={_differing_element_count(first, second)} "
        f"first_diff_coordinate={first_diff} first_value={first_value} second_value={second_value}"
    )


@pytest.mark.parametrize(
    "tokens",
    _TAIL_TEST_TOKENS,
    ids=lambda tokens: f"tokens_{tokens}_remainder_{tokens % _CHUNK_SIZE}",
)
@pytest.mark.parametrize("gate_mode", ["external_gate", "raw_gate"])
@pytest.mark.parametrize("metadata_mode", ["dense", "varlen"])
@torch.inference_mode()
def test_kimi_k3_chunk_kda_bf16_tail_is_deterministic(tokens: int, gate_mode: str, metadata_mode: str):
    if not hasattr(torch.ops._C_ascend, "chunk_kda_fwd"):
        pytest.skip("requires the fused chunk KDA AscendC operator")

    inputs = _build_inputs(tokens)
    # Create both input sets before the first custom-op call so a write past an
    # input boundary cannot contaminate the second invocation's tensors.
    first_inputs = inputs.clone()
    second_inputs = inputs.clone()
    first = _snapshot_outputs(_run_chunk_kda(first_inputs, gate_mode, metadata_mode))
    second = _snapshot_outputs(_run_chunk_kda(second_inputs, gate_mode, metadata_mode))

    problems = [
        problem
        for name, first_output, second_output in zip(_OUTPUT_NAMES, first[:11], second[:11])
        if (problem := _describe_difference(name, first_output, second_output)) is not None
    ]
    assert not problems, (
        f"Kimi-K3 chunk KDA is not deterministic for tokens={tokens}, "
        f"remainder={tokens % _CHUNK_SIZE}, gate_mode={gate_mode}, metadata_mode={metadata_mode}:\n"
        + "\n".join(problems)
    )
