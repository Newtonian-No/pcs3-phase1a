import torch

from temporal_mamba.generalized_coordinates import (
    CoordinateBatch,
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


def test_aligned_errors_use_h_t_to_predict_target_t_plus_one():
    predictor = GeneralizedCoordinatePredictor(hidden_dim=1, signal_dim=1)
    with torch.no_grad():
        for head in predictor.heads:
            head.weight.zero_()
            head.bias.zero_()
        predictor.heads[0].weight.fill_(2.0)
    hidden = torch.tensor([[[-1.0], [1.0], [-1.0], [1.0]]])
    coordinates = causal_coordinate_targets(
        torch.tensor([[[0.0], [2.0], [5.0], [9.0]]])
    )

    errors, _ = aligned_coordinate_errors(hidden, coordinates, predictor)

    normalized_hidden = torch.rsqrt(torch.tensor(1.0 + 1e-6))
    raw = torch.tensor(
        [
            2.0 + 2.0 * normalized_hidden,
            5.0 - 2.0 * normalized_hidden,
            9.0 + 2.0 * normalized_hidden,
        ]
    )
    expected = raw / raw.square().mean().sqrt()
    torch.testing.assert_close(errors[0, 1:, 0, 0], expected)


def test_alignment_detaches_inputs_but_trains_predictor():
    predictor = GeneralizedCoordinatePredictor(hidden_dim=4, signal_dim=2)
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    signal = torch.randn(2, 5, 2, requires_grad=True)
    coordinates = causal_coordinate_targets(signal)

    errors, _ = aligned_coordinate_errors(hidden, coordinates, predictor)
    errors.sum().backward()

    assert hidden.grad is None
    assert signal.grad is None
    assert all(parameter.grad is not None for parameter in predictor.parameters())


def test_aligned_errors_normalize_each_order_independently():
    predictor = GeneralizedCoordinatePredictor(hidden_dim=4, signal_dim=2)
    hidden = torch.randn(3, 6, 4)
    coordinates = causal_coordinate_targets(torch.randn(3, 6, 2))

    errors, valid = aligned_coordinate_errors(hidden, coordinates, predictor)

    expanded_valid = valid.expand_as(errors)
    for order in range(3):
        order_values = errors[:, :, order][expanded_valid[:, :, order]]
        torch.testing.assert_close(
            order_values.square().mean().sqrt(), torch.tensor(1.0)
        )


def test_alignment_accepts_broadcastable_coordinate_mask():
    coordinates = causal_coordinate_targets(torch.randn(2, 5, 1))
    broadcast_coordinates = CoordinateBatch(
        targets=coordinates.targets,
        mask=coordinates.mask[:1],
    )
    predictor = GeneralizedCoordinatePredictor(hidden_dim=4, signal_dim=1)

    errors, valid = aligned_coordinate_errors(
        torch.randn(2, 5, 4), broadcast_coordinates, predictor
    )

    assert errors.shape == (2, 5, 3, 1)
    assert valid.shape == (2, 5, 3, 1)
    torch.testing.assert_close(valid[0], valid[1])


def test_order_mask_keeps_fixed_flat_dimension():
    errors = torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).view(2, 5, 3, 4)
    for order in (1, 2, 3):
        flat = select_active_orders(errors, order)
        assert flat.shape == (2, 5, 12)
        assert torch.count_nonzero(flat[..., order * 4 :]) == 0


def test_controls_are_deterministic_and_match_statistics():
    error = torch.randn(8, 9, 12)
    valid = torch.ones(8, 9, 3, 1, dtype=torch.bool)
    shuffled_a = controlled_error(error, "gc_k3_shuffled", valid=valid, seed=17)
    shuffled_b = controlled_error(error, "gc_k3_shuffled", valid=valid, seed=17)
    torch.testing.assert_close(shuffled_a, shuffled_b)
    torch.testing.assert_close(shuffled_a.mean((0, 1)), error.mean((0, 1)))
    noise = controlled_error(error, "gc_k3_noise", valid=valid, seed=17)
    torch.testing.assert_close(
        noise.mean((0, 1)), error.mean((0, 1)), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        noise.std((0, 1)), error.std((0, 1)), atol=1e-4, rtol=1e-4
    )


def test_noise_matches_valid_statistics_and_keeps_invalid_entries_zero():
    error = torch.randn(4, 5, 6)
    valid = torch.ones(4, 5, 3, 1, dtype=torch.bool)
    valid[:, 0] = False
    valid[:, :, 2] = False
    valid[0, 2, 1] = False
    flat_valid = valid.expand(4, 5, 3, 2).reshape(4, 5, 6)
    error = torch.where(flat_valid, error, 0.0)

    noise = controlled_error(error, "gc_k3_noise", valid=valid, seed=23)

    assert torch.count_nonzero(noise.masked_select(~flat_valid)) == 0
    for channel in range(error.shape[-1]):
        channel_valid = flat_valid[..., channel]
        if not channel_valid.any():
            continue
        torch.testing.assert_close(
            noise[..., channel][channel_valid].mean(),
            error[..., channel][channel_valid].mean(),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            noise[..., channel][channel_valid].std(),
            error[..., channel][channel_valid].std(),
            atol=1e-4,
            rtol=1e-4,
        )


def test_noise_is_finite_and_deterministic_with_one_valid_sample():
    error = torch.tensor([[[2.0, -3.0, 0.0, 0.0, 0.0, 0.0]]])
    valid = torch.tensor([[[[True], [False], [False]]]])

    first = controlled_error(error, "gc_k3_noise", valid=valid, seed=31)
    second = controlled_error(error, "gc_k3_noise", valid=valid, seed=31)

    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, error)
