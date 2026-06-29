"""Tests for the Tensor class in micro_torch."""

import pytest
from micro_torch import Tensor


class TestTensorInitialization:
    def test_leaf_tensor_stores_data_and_defaults(self):
        tensor = Tensor(5)
        assert tensor.data == 5
        assert tensor.parents == []
        assert tensor.op is None
        assert tensor.grad == 0.0

    def test_tensor_can_be_created_with_metadata(self):
        parent_a = Tensor(2)
        parent_b = Tensor(3)
        tensor = Tensor(5, parents=[parent_a, parent_b], op='+')
        assert tensor.data == 5
        assert tensor.parents == [parent_a, parent_b]
        assert tensor.op == '+'


def test_repr_uses_tensor_data():
    assert repr(Tensor(3.14)) == "Tensor(3.14)"


def test_ensure_tensor_wraps_scalar_values():
    tensor = Tensor(5)
    assert tensor._ensure_tensor(tensor) is tensor
    assert isinstance(tensor._ensure_tensor(7), Tensor)
    assert tensor._ensure_tensor(7.5).data == 7.5


def test_addition_computes_sum_and_tracks_parents():
    left = Tensor(2)
    right = Tensor(3)
    result = left + right

    assert result.data == 5
    assert result.op == '+'
    assert result.parents == [left, right]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(2, 3, 5), (2, 2.5, 4.5), (3, -1, 2)],
)
def test_addition_supports_scalars(left, right, expected):
    assert (Tensor(left) + right).data == expected
    assert (left + Tensor(right)).data == expected


def test_subtraction_returns_expected_value_and_graph():
    left = Tensor(10)
    right = Tensor(3)
    result = left - right

    assert result.data == 7
    assert result.op == '-'
    assert result.parents[0] is left
    assert result.parents[1].data == -3
    assert result.parents[1].op == 'neg'


def test_multiplication_and_reverse_multiplication():
    assert (Tensor(5) * 3).data == 15
    assert (3 * Tensor(5)).data == 15
    assert (Tensor(2) * Tensor(4)).data == 8


def test_division_and_reverse_division():
    assert (Tensor(10) / 2).data == 5.0
    assert (10 / Tensor(2)).data == 5.0
    assert (Tensor(10) / Tensor(2)).data == 5.0


def test_division_by_zero_raises_error():
    with pytest.raises(ZeroDivisionError):
        Tensor(10) / 0


@pytest.mark.parametrize(("base", "exp", "expected"), [(2, 3, 8), (4, 0.5, 2.0), (5, 0, 1)])
def test_power_supports_scalar_exponents(base, exp, expected):
    assert (Tensor(base) ** exp).data == expected


def test_power_with_tensor_exponent_is_not_supported():
    with pytest.raises(AssertionError, match="only scalar exponents"):
        Tensor(2) ** Tensor(3)


def test_negation_flips_sign():
    assert (-Tensor(5)).data == -5
    assert (-Tensor(-5)).data == 5


def test_complex_expression_uses_the_graph():
    left = Tensor(2)
    right = Tensor(3)
    result = (left + 5) * (right - 1)

    assert result.data == 14
    assert result.op == '*'
    assert len(result.parents) == 2


def test_operations_do_not_mutate_original_tensors():
    tensor = Tensor(5)
    original_data = tensor.data

    _ = tensor + 3
    _ = tensor * 2
    _ = -tensor

    assert tensor.data == original_data
