import pytest
import torch

from temporal_mamba.config import ModelConfig
from temporal_mamba.model import NextStepPredictor, TemporalMambaModel, aligned_errors


def make_tiny_model(num_outputs=1):
    config = ModelConfig(
        d_model=8,
        d_state=4,
        n_layers=2,
        expand=1,
        dt_min=1e-3,
        dt_max=1e-1,
        alpha_max=1.38629436112,
        dropout=0.0,
    )
    return TemporalMambaModel(
        input_dim=20,
        signal_dim=4,
        num_outputs=num_outputs,
        model_config=config,
    )


@pytest.mark.parametrize(
    ("variant", "passes", "uses_error"),
    [
        ("vanilla", 1, False),
        ("two_pass", 2, False),
        ("error_inject", 2, True),
        ("error_aux", 2, True),
        ("time_shuffle", 2, True),
        ("time_reverse", 2, True),
    ],
)
def test_variant_forward_contract(variant, passes, uses_error):
    torch.manual_seed(10)
    model = make_tiny_model().eval()
    features = torch.randn(3, 12, 20)
    signal = torch.randn(3, 12, 4)
    output = model(features, signal, variant=variant, return_diagnostics=True)
    assert output.logits.shape == (3, 1)
    assert output.pass_count == passes
    assert output.uses_error is uses_error
    assert (output.position_error is not None) is uses_error
    assert (output.velocity_error is not None) is uses_error
    assert bool(output.diagnostics["finite"])


def test_aligned_errors_detach_hidden_and_targets_but_train_predictor():
    torch.manual_seed(11)
    predictor = NextStepPredictor(hidden_dim=5, signal_dim=2)
    hidden = torch.randn(2, 6, 5, requires_grad=True)
    signal = torch.randn(2, 6, 2, requires_grad=True)
    with torch.no_grad():
        expected_x, expected_v = predictor(hidden.detach())
    position, velocity = aligned_errors(hidden, signal, predictor)
    assert torch.equal(position[:, 0], torch.zeros_like(position[:, 0]))
    assert torch.equal(velocity[:, 0], torch.zeros_like(velocity[:, 0]))
    torch.testing.assert_close(position[:, 1:], signal[:, 1:].detach() - expected_x[:, :-1])
    expected_delta = signal[:, 1:].detach() - signal[:, :-1].detach()
    torch.testing.assert_close(velocity[:, 1:], expected_delta - expected_v[:, :-1])
    (position.square().mean() + velocity.square().mean()).backward()
    assert hidden.grad is None
    assert signal.grad is None
    assert all(parameter.grad is not None for parameter in predictor.parameters())


def test_one_and_two_pass_reuse_the_same_encoder_modules():
    torch.manual_seed(12)
    model = make_tiny_model().eval()
    calls = []
    handle = model.layers[0].register_forward_hook(lambda *_: calls.append(1))
    features = torch.randn(2, 10, 20)
    signal = torch.randn(2, 10, 4)
    model(features, signal, variant="vanilla")
    assert len(calls) == 1
    calls.clear()
    model(features, signal, variant="two_pass")
    assert len(calls) == 2
    handle.remove()


@pytest.mark.parametrize("length", [64, 128, 256])
def test_lengths_are_finite_and_dt_is_bounded(length):
    torch.manual_seed(13)
    model = make_tiny_model().eval()
    for layer in model.layers:
        layer.ssm.alpha_raw.data.fill_(50)
    features = torch.randn(1, length, 20)
    signal = torch.randn(1, length, 4) * 1e4
    output = model(features, signal, variant="error_aux", return_diagnostics=True)
    assert torch.isfinite(output.logits).all()
    assert float(output.diagnostics["dt_min"].detach()) >= 1e-3
    assert float(output.diagnostics["dt_max"].detach()) <= 1e-1
    assert bool(output.diagnostics["finite"])


def test_multiclass_head_shape():
    model = make_tiny_model(num_outputs=6).eval()
    output = model(
        torch.randn(4, 9, 20),
        torch.randn(4, 9, 4),
        variant="vanilla",
    )
    assert output.logits.shape == (4, 6)


def test_rejects_misaligned_inputs_and_unknown_variant():
    model = make_tiny_model()
    features = torch.randn(2, 8, 20)
    signal = torch.randn(2, 7, 4)
    with pytest.raises(ValueError, match="align"):
        model(features, signal, variant="vanilla")
    with pytest.raises(ValueError, match="variant"):
        model(features, torch.randn(2, 8, 4), variant="future")
