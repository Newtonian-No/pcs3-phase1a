import torch

from temporal_mamba.generalized_coordinates import (
    GeneralizedCoordinatePredictor,
    aligned_coordinate_errors,
    causal_coordinate_targets,
    controlled_error,
    select_active_orders,
)


def test_causal_coordinates_use_only_present_and_past():
    signal = torch.tensor([[[0.0], [1.0], [4.0], [9.0]]])
    batch = causal_coordinate_targets(signal)
    assert batch.targets.shape == (1, 4, 3, 1)
    torch.testing.assert_close(
        batch.targets[0, :, 0, 0], torch.tensor([0.0, 1.0, 4.0, 9.0])
    )
    torch.testing.assert_close(
        batch.targets[0, :, 1, 0], torch.tensor([0.0, 1.0, 3.0, 5.0])
    )
    torch.testing.assert_close(
        batch.targets[0, :, 2, 0], torch.tensor([0.0, 0.0, 2.0, 2.0])
    )
    assert batch.mask[0, :, 0, 0].tolist() == [True, True, True, True]
    assert batch.mask[0, :, 1, 0].tolist() == [False, True, True, True]
    assert batch.mask[0, :, 2, 0].tolist() == [False, False, True, True]


def test_future_mutation_does_not_change_past_coordinates():
    first = torch.randn(2, 8, 3)
    second = first.clone()
    second[:, 6:] += 100
    a = causal_coordinate_targets(first)
    b = causal_coordinate_targets(second)
    torch.testing.assert_close(a.targets[:, :6], b.targets[:, :6])


def test_aligned_errors_shift_predictions_one_step():
    predictor = GeneralizedCoordinatePredictor(hidden_dim=4, signal_dim=1)
    hidden = torch.randn(2, 5, 4)
    coordinates = causal_coordinate_targets(torch.randn(2, 5, 1))
    errors, valid = aligned_coordinate_errors(hidden, coordinates, predictor)
    assert errors.shape == (2, 5, 3, 1)
    assert not valid[:, 0].any()
    assert valid[:, 1, 0].all()
    assert not valid[:, 1, 2].any()


def test_order_mask_keeps_fixed_flat_dimension():
    errors = torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).view(2, 5, 3, 4)
    for order in (1, 2, 3):
        flat = select_active_orders(errors, order)
        assert flat.shape == (2, 5, 12)
        assert torch.count_nonzero(flat[..., order * 4 :]) == 0


def test_controls_are_deterministic_and_match_statistics():
    error = torch.randn(8, 9, 12)
    shuffled_a = controlled_error(error, "gc_k3_shuffled", seed=17)
    shuffled_b = controlled_error(error, "gc_k3_shuffled", seed=17)
    torch.testing.assert_close(shuffled_a, shuffled_b)
    torch.testing.assert_close(shuffled_a.mean((0, 1)), error.mean((0, 1)))
    noise = controlled_error(error, "gc_k3_noise", seed=17)
    torch.testing.assert_close(
        noise.mean((0, 1)), error.mean((0, 1)), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        noise.std((0, 1)), error.std((0, 1)), atol=1e-4, rtol=1e-4
    )
