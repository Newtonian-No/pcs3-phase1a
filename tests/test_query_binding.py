import pytest
import torch

from temporal_mamba.query_binding import BoundedQueryFiLM, TemporalQueryBinder


def _make_query(
    *,
    event_a: int,
    event_b: int | None,
    p0: int,
    p1: int,
    p2: int = 0,
    seq_len: int,
    family: int = 0,
    event_dim: int = 8,
) -> torch.Tensor:
    query = torch.zeros(1, 6 + 2 * event_dim + 3)
    query[0, family] = 1.0
    query[0, 6 + event_a] = 1.0
    if event_b is not None:
        query[0, 6 + event_dim + event_b] = 1.0
    query[0, -3:] = torch.tensor([p0, p1, p2]) / max(seq_len - 1, 1)
    return query


def test_binder_extracts_exact_a_b_and_relative_time():
    signal = torch.zeros(1, 5, 8)
    signal[0, 1, 3] = 1.0
    signal[0, 4, 6] = 1.0
    query = _make_query(event_a=3, event_b=6, p0=1, p1=4, seq_len=5)

    bound = TemporalQueryBinder(event_dim=8)(signal, query)

    time = torch.linspace(0.0, 1.0, 5)
    torch.testing.assert_close(bound.sequence[0, :, 0], signal[0, :, 3])
    torch.testing.assert_close(bound.sequence[0, :, 1], signal[0, :, 6])
    torch.testing.assert_close(bound.sequence[0, :, 2], time)
    torch.testing.assert_close(bound.sequence[0, :, 3], time - 0.25)
    torch.testing.assert_close(bound.sequence[0, :, 4], time - 1.0)
    torch.testing.assert_close(bound.prediction_signal, bound.sequence[:, :, :2])
    assert bound.sequence.shape == (1, 5, 5)
    assert bound.condition.shape == (1, 9)
    assert bound.prediction_signal.shape == (1, 5, 2)


def test_binder_accepts_missing_b_as_an_all_zero_stream():
    signal = torch.randn(2, 7, 8)
    query = _make_query(event_a=2, event_b=None, p0=1, p1=5, seq_len=7).expand(2, -1)

    bound = TemporalQueryBinder(event_dim=8)(signal, query)

    torch.testing.assert_close(bound.sequence[:, :, 1], torch.zeros(2, 7))


def test_binder_is_invariant_to_joint_channel_permutation():
    generator = torch.Generator().manual_seed(17)
    signal = torch.randn(4, 17, 8, generator=generator)
    query = torch.cat(
        [
            _make_query(
                event_a=index,
                event_b=7 - index,
                p0=2,
                p1=12,
                seq_len=17,
                family=index,
            )
            for index in range(4)
        ],
        dim=0,
    )
    permutation = torch.tensor([6, 2, 0, 7, 4, 1, 5, 3])
    permuted_signal = signal[:, :, permutation]
    permuted_query = query.clone()
    permuted_query[:, 6:14] = query[:, 6:14][:, permutation]
    permuted_query[:, 14:22] = query[:, 14:22][:, permutation]
    binder = TemporalQueryBinder(event_dim=8)

    original = binder(signal, query)
    permuted = binder(permuted_signal, permuted_query)

    torch.testing.assert_close(original.sequence, permuted.sequence)
    torch.testing.assert_close(original.condition, permuted.condition)
    torch.testing.assert_close(original.prediction_signal, permuted.prediction_signal)


@pytest.mark.parametrize(
    ("signal", "query", "match"),
    [
        (torch.zeros(2, 8), torch.zeros(2, 25), "signal"),
        (torch.zeros(2, 5, 8), torch.zeros(25), "query"),
        (torch.zeros(2, 5, 8), torch.zeros(3, 25), "batch"),
        (torch.zeros(2, 5, 7), torch.zeros(2, 25), "event_dim"),
        (torch.zeros(2, 5, 8), torch.zeros(2, 24), "query"),
    ],
)
def test_binder_rejects_invalid_shapes(signal, query, match):
    with pytest.raises(ValueError, match=match):
        TemporalQueryBinder(event_dim=8)(signal, query)


def test_binder_rejects_nonfinite_and_non_one_hot_queries():
    signal = torch.zeros(1, 5, 8)
    valid = _make_query(event_a=3, event_b=6, p0=1, p1=4, seq_len=5)
    invalid_cases = []
    nonfinite = valid.clone()
    nonfinite[0, -1] = torch.nan
    invalid_cases.append(nonfinite)
    missing_a = valid.clone()
    missing_a[:, 6:14] = 0.0
    invalid_cases.append(missing_a)
    multiple_b = valid.clone()
    multiple_b[0, 14] = 1.0
    invalid_cases.append(multiple_b)
    missing_family = valid.clone()
    missing_family[:, :6] = 0.0
    invalid_cases.append(missing_family)

    binder = TemporalQueryBinder(event_dim=8)
    for query in invalid_cases:
        with pytest.raises(ValueError):
            binder(signal, query)


def test_film_starts_identity_and_stays_bounded_after_updates():
    film = BoundedQueryFiLM(condition_dim=9, n_layers=4, d_model=16)
    condition = torch.randn(3, 9) * 1e6

    scale, shift = film(condition)

    torch.testing.assert_close(scale, torch.ones_like(scale))
    torch.testing.assert_close(shift, torch.zeros_like(shift))
    with torch.no_grad():
        film.projection.weight.fill_(100.0)
        film.projection.bias.copy_(torch.linspace(-100.0, 100.0, film.projection.bias.numel()))
    scale, shift = film(condition)
    assert scale.shape == (3, 4, 16)
    assert shift.shape == (3, 4, 16)
    assert torch.all(scale >= 0.75) and torch.all(scale <= 1.25)
    assert torch.all(shift >= -0.25) and torch.all(shift <= 0.25)
    assert torch.isfinite(scale).all() and torch.isfinite(shift).all()


def test_film_rejects_wrong_condition_shape():
    film = BoundedQueryFiLM(condition_dim=9, n_layers=2, d_model=4)
    with pytest.raises(ValueError, match="condition"):
        film(torch.zeros(3, 8))

