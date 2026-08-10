from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = ROOT / "csrc/attention/chunk_kda_fwd"


def test_chunk_kda_fwd_uses_one_physical_l0_entry():
    kernel_entry = (OP_ROOT / "op_kernel/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    l0_launcher = (OP_ROOT / "op_host/op_api/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(encoding="utf-8")

    assert kernel_entry.count('extern "C" __global__ __aicore__ void chunk_kda_fwd(') == 1
    assert l0_launcher.count("ADD_TO_LAUNCHER_LIST_AICORE(") == 1
    assert aclnn.count("l0op::KdaChunkForward(") == 1
    for stage in ("ChunkKdaFwdPrepare", "ChunkKdaFwdPostWu", "ChunkKdaFwdFinalize"):
        assert stage not in l0_launcher
        assert stage not in aclnn


def test_chunk_kda_fwd_declares_internal_kernel_dependencies():
    cmake = (OP_ROOT / "op_host/CMakeLists.txt").read_text(encoding="utf-8")
    common = (OP_ROOT / "op_kernel/chunk_kda_fwd_common.h").read_text(encoding="utf-8")

    assert "attention/kda_gate_cumsum" in cmake
    assert "moe/chunk_gated_delta_rule_fwd_h" in cmake
    assert "../../kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h" in common
    assert "chunk_gated_delta_rule_fwd_h/op_kernel" in common


def test_chunk_kda_fwd_binding_exposes_latest_fused_contract():
    binding = (ROOT / "csrc/torch_binding.cpp").read_text(encoding="utf-8")
    meta = (ROOT / "csrc/torch_binding_meta.cpp").read_text(encoding="utf-8")
    schema = next(line for line in binding.splitlines() if '"chunk_kda_fwd(Tensor q' in line)

    for argument in (
        "use_gate_in_kernel",
        "A_log",
        "dt_bias",
        "disable_recompute",
        "return_intermediate_states",
        "state_v_first",
    ):
        assert argument in schema
    assert "return_intermediate=" not in schema
    assert "transpose_state_layout=" not in schema
    assert "c10::optional<at::Tensor>>\nchunk_kda_fwd(" in binding
    assert "c10::optional<at::Tensor>>\nchunk_kda_fwd_meta(" in meta
