import pytest
import torch

from temporal_mamba.ssm import (
    DirectSelectiveSSM,
    TemporalMambaBlock,
    direct_selective_scan,
    inverse_softplus,
)


def reference_scan(u, dt, a_log, b, c, d_skip):
    u32, dt32 = u.float(), dt.float()
    a = -torch.exp(a_log.float())
    h = torch.zeros(u.size(0), u.size(2), a.size(1), dtype=torch.float32, device=u.device)
    ys = []
    for t in range(u.size(1)):
        a_bar = torch.exp(dt32[:, t].unsqueeze(-1) * a)
        b_bar = dt32[:, t].unsqueeze(-1) * b[:, t].float().unsqueeze(1)
        h = a_bar * h + b_bar * u32[:, t].unsqueeze(-1)
        ys.append((h * c[:, t].float().unsqueeze(1)).sum(-1) + d_skip.float() * u32[:, t])
    return torch.stack(ys, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_direct_scan_matches_independent_reference_and_gradients(dtype):
    torch.manual_seed(3)
    shapes = ((2, 5, 3), (2, 5, 3), (3, 2), (2, 5, 2), (2, 5, 2), (3,))
    implementation = [torch.randn(shape, dtype=dtype, requires_grad=True) for shape in shapes]
    reference = [tensor.detach().clone().requires_grad_(True) for tensor in implementation]
    implementation[1].data.abs_().mul_(0.05).add_(0.001)
    reference[1].data.copy_(implementation[1].data)
    implementation[2].data.abs_()
    reference[2].data.copy_(implementation[2].data)

    actual = direct_selective_scan(*implementation)
    expected = reference_scan(*reference)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    weights = torch.linspace(0.2, 1.2, actual.numel(), device=actual.device).reshape_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for actual_tensor, expected_tensor in zip(implementation, reference):
        assert actual_tensor.grad is not None
        assert expected_tensor.grad is not None
        torch.testing.assert_close(actual_tensor.grad, expected_tensor.grad, rtol=1e-5, atol=1e-5)


def test_inverse_softplus_round_trip():
    values = torch.logspace(-4, 1, 30)
    torch.testing.assert_close(torch.nn.functional.softplus(inverse_softplus(values)), values)


def test_zero_error_scale_matches_base_dt_and_extremes_are_bounded():
    torch.manual_seed(4)
    layer = DirectSelectiveSSM(16, d_state=4, error_dim=6, dt_min=1e-3, dt_max=1e-1)
    u = torch.randn(2, 32, 16)
    base = layer.compute_dt(u, error=None)
    zero_scaled = layer.compute_dt(u, error=torch.randn(2, 32, 6) * 1e8)
    assert torch.equal(base, zero_scaled)
    layer.alpha_raw.data.fill_(100)
    bounded = layer.compute_dt(u, error=torch.randn(2, 32, 6) * 1e8)
    assert float(bounded.detach().min()) >= 1e-3
    assert float(bounded.detach().max()) <= 1e-1
    assert torch.equal(layer.a_log[0], torch.log(torch.arange(1, 5, dtype=torch.float32)))
    assert torch.equal(layer.d_skip, torch.ones(16))
    assert layer.dt_rank == 1


def test_ssm_diagnostics_and_float64_input_are_finite():
    torch.manual_seed(5)
    layer = DirectSelectiveSSM(9, d_state=3, error_dim=4, dt_min=1e-3, dt_max=1e-1)
    u = torch.randn(2, 17, 9, dtype=torch.float64)
    error = torch.randn(2, 17, 4, dtype=torch.float64)
    output, diagnostics = layer(u, error=error, return_diagnostics=True)
    assert output.shape == u.shape
    assert output.dtype == torch.float32
    assert bool(diagnostics.finite)
    assert float(diagnostics.dt_min.detach()) >= 1e-3
    assert float(diagnostics.dt_max.detach()) <= 1e-1
    assert float(diagnostics.error_rms.detach()) > 0


def test_temporal_block_is_causal_and_shape_preserving():
    torch.manual_seed(6)
    block = TemporalMambaBlock(
        d_model=8,
        d_state=4,
        expand=2,
        error_dim=6,
        dt_min=1e-3,
        dt_max=1e-1,
        alpha_max=1.38629436112,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 15, 8)
    error = torch.randn(2, 15, 6)
    changed = x.clone()
    changed[:, 10:] += 100
    original_output, diagnostics = block(x, error=error, return_diagnostics=True)
    changed_output = block(changed, error=error, return_diagnostics=False)
    assert original_output.shape == x.shape
    torch.testing.assert_close(original_output[:, :10], changed_output[:, :10], rtol=0, atol=1e-6)
    assert bool(diagnostics.finite)
