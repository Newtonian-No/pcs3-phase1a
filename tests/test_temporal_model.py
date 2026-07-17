import pytest
import torch

from temporal_mamba.config import GC_MATRIX_VARIANTS, ModelConfig
from temporal_mamba.generalized_coordinates import (
    GeneralizedCoordinatePredictor,
    causal_coordinate_targets,
)
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


def make_tiny_v2_model(num_outputs=1):
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
        input_dim=34,
        signal_dim=8,
        num_outputs=num_outputs,
        model_config=config,
        input_mode="query_bound",
    )


def make_v2_query(batch: int, length: int) -> torch.Tensor:
    query = torch.zeros(batch, 25)
    for index in range(batch):
        query[index, index % 6] = 1.0
        query[index, 6 + index % 8] = 1.0
        query[index, 14 + (index + 3) % 8] = 1.0
    query[:, -3:] = torch.tensor([2, length - 3, 1]) / (length - 1)
    return query


def permute_v2_query(query: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    permuted = query.clone()
    permuted[:, 6:14] = query[:, 6:14][:, permutation]
    permuted[:, 14:22] = query[:, 14:22][:, permutation]
    return permuted


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


def test_query_bound_model_is_channel_permutation_invariant():
    torch.manual_seed(14)
    model = make_tiny_v2_model().eval()
    signal = torch.randn(3, 24, 8)
    query = make_v2_query(batch=3, length=24)
    permutation = torch.tensor([3, 7, 1, 6, 0, 5, 2, 4])

    first = model(signal, signal, query=query, variant="vanilla")
    second = model(
        signal[:, :, permutation],
        signal[:, :, permutation],
        query=permute_v2_query(query, permutation),
        variant="vanilla",
    )

    torch.testing.assert_close(first.logits, second.logits)


def test_v2_readout_and_error_target_use_bound_streams():
    torch.manual_seed(15)
    model = make_tiny_v2_model().eval()
    signal = torch.randn(2, 32, 8)
    query = make_v2_query(batch=2, length=32)

    output = model(
        signal,
        signal,
        query=query,
        variant="error_aux",
        return_diagnostics=True,
    )

    assert model.classifier.in_features == 3 * model.model_config.d_model
    assert output.logits.shape == (2, 1)
    assert output.position_error is not None and output.position_error.shape == (2, 32, 2)
    assert output.velocity_error is not None and output.velocity_error.shape == (2, 32, 2)
    assert bool(output.diagnostics["finite"])


def test_query_bound_model_requires_query_but_legacy_contract_is_unchanged():
    bound_model = make_tiny_v2_model().eval()
    signal = torch.randn(2, 12, 8)
    with pytest.raises(ValueError, match="requires query"):
        bound_model(signal, signal, variant="vanilla")

    legacy_model = make_tiny_model().eval()
    output = legacy_model(
        torch.randn(2, 12, 20),
        torch.randn(2, 12, 4),
        variant="vanilla",
    )
    assert output.logits.shape == (2, 1)
    assert legacy_model.classifier.in_features == legacy_model.model_config.d_model


def test_rejects_misaligned_inputs_and_unknown_variant():
    model = make_tiny_model()
    features = torch.randn(2, 8, 20)
    signal = torch.randn(2, 7, 4)
    with pytest.raises(ValueError, match="align"):
        model(features, signal, variant="vanilla")
    with pytest.raises(ValueError, match="variant"):
        model(features, torch.randn(2, 8, 4), variant="future")


@pytest.mark.parametrize(
    ("variant", "order"),
    [
        ("gc_k1", 1),
        ("gc_k2", 2),
        ("gc_k3", 3),
        ("gc_k3_shuffled", 3),
        ("gc_k3_noise", 3),
    ],
)
def test_gc_forward_has_fixed_error_width(variant, order):
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
    model = TemporalMambaModel(
        input_dim=7,
        signal_dim=6,
        num_outputs=3,
        model_config=config,
        generalized_coordinates=True,
    )
    signal = torch.randn(4, 16, 6)
    time = torch.linspace(0, 1, 16).view(1, 16, 1).expand(4, -1, -1)
    features = torch.cat((signal, time), -1)
    coordinates = causal_coordinate_targets(signal)
    output = model(
        features,
        signal,
        variant=variant,
        coordinate_targets=coordinates.targets,
        coordinate_mask=coordinates.mask,
        error_control_seed=99,
    )
    assert output.coordinate_errors.shape == (4, 16, 18)
    assert output.coordinate_mask.shape == (4, 16, 3, 1)
    assert output.gc_order == order
    assert output.pass_count == 2


def test_all_gc_variants_have_identical_parameter_count():
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
    counts = []
    for _variant in GC_MATRIX_VARIANTS:
        model = TemporalMambaModel(
            input_dim=7,
            signal_dim=6,
            num_outputs=3,
            model_config=config,
            generalized_coordinates=True,
        )
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        assert isinstance(model.predictor, GeneralizedCoordinatePredictor)
        assert not any(isinstance(module, NextStepPredictor) for module in model.modules())
        assert all(layer.ssm.error_dim == 18 for layer in model.layers)
    assert len(set(counts)) == 1


@pytest.mark.parametrize(("variant", "passes"), [("vanilla", 1), ("two_pass", 2)])
def test_gc_baselines_build_gc_modules_without_computing_errors(variant, passes):
    model = TemporalMambaModel(
        input_dim=20,
        signal_dim=4,
        num_outputs=3,
        model_config=make_tiny_model().model_config,
        generalized_coordinates=True,
    )
    output = model(
        torch.randn(2, 8, 20),
        torch.randn(2, 8, 4),
        variant=variant,
    )
    assert output.pass_count == passes
    assert output.coordinate_errors is None
    assert output.coordinate_mask is None
    assert output.gc_order == 0


def test_gc_error_variants_require_coordinate_targets_and_mask():
    model = TemporalMambaModel(
        input_dim=20,
        signal_dim=4,
        num_outputs=3,
        model_config=make_tiny_model().model_config,
        generalized_coordinates=True,
    )
    features = torch.randn(2, 8, 20)
    signal = torch.randn(2, 8, 4)
    coordinates = causal_coordinate_targets(signal)
    with pytest.raises(ValueError, match="coordinate_targets"):
        model(features, signal, variant="gc_k1", coordinate_mask=coordinates.mask)
    with pytest.raises(ValueError, match="coordinate_mask"):
        model(features, signal, variant="gc_k1", coordinate_targets=coordinates.targets)


def test_gc_controls_replace_only_the_injected_error():
    torch.manual_seed(21)
    model = TemporalMambaModel(
        input_dim=20,
        signal_dim=4,
        num_outputs=3,
        model_config=make_tiny_model().model_config,
        generalized_coordinates=True,
    ).eval()
    features = torch.randn(3, 8, 20)
    signal = torch.randn(3, 8, 4)
    coordinates = causal_coordinate_targets(signal)
    common = dict(
        coordinate_targets=coordinates.targets,
        coordinate_mask=coordinates.mask,
        error_control_seed=7,
    )
    clean = model(features, signal, variant="gc_k3", **common)
    shuffled = model(features, signal, variant="gc_k3_shuffled", **common)
    noise = model(features, signal, variant="gc_k3_noise", **common)
    torch.testing.assert_close(shuffled.coordinate_errors, clean.coordinate_errors)
    torch.testing.assert_close(noise.coordinate_errors, clean.coordinate_errors)


def test_explicit_false_gc_flag_preserves_legacy_model_and_forward():
    torch.manual_seed(22)
    implicit = make_tiny_model().eval()
    torch.manual_seed(22)
    explicit = TemporalMambaModel(
        input_dim=20,
        signal_dim=4,
        num_outputs=1,
        model_config=implicit.model_config,
        generalized_coordinates=False,
    ).eval()
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for name, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name])
    features = torch.randn(2, 8, 20)
    signal = torch.randn(2, 8, 4)
    first = implicit(features, signal, variant="error_aux")
    second = explicit(features, signal, variant="error_aux")
    torch.testing.assert_close(first.logits, second.logits)
    torch.testing.assert_close(first.position_error, second.position_error)
    torch.testing.assert_close(first.velocity_error, second.velocity_error)
