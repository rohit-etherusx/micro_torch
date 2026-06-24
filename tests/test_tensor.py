"""Comprehensive test suite for the Tensor class."""

import pytest
from micro_torch import Tensor


class TestTensorInitialization:
    """Test Tensor initialization and basic properties."""

    def test_init_with_integer(self):
        """Test Tensor initialization with integer."""
        t = Tensor(5)
        assert t.data == 5
        assert t.parents == []
        assert t.op is None

    def test_init_with_float(self):
        """Test Tensor initialization with float."""
        t = Tensor(3.14)
        assert t.data == 3.14
        assert t.parents == []
        assert t.op is None

    def test_init_with_zero(self):
        """Test Tensor initialization with zero."""
        t = Tensor(0)
        assert t.data == 0
        assert t.parents == []

    def test_init_with_negative(self):
        """Test Tensor initialization with negative value."""
        t = Tensor(-10)
        assert t.data == -10

    def test_init_with_parents_and_op(self):
        """Test Tensor initialization with parents and operation."""
        parent1 = Tensor(5)
        parent2 = Tensor(3)
        t = Tensor(8, parents=[parent1, parent2], op='+')
        assert t.data == 8
        assert t.parents == [parent1, parent2]
        assert t.op == '+'


class TestTensorRepr:
    """Test string representation of Tensor."""

    def test_repr_integer(self):
        """Test __repr__ with integer."""
        t = Tensor(5)
        assert repr(t) == "Tensor(5)"

    def test_repr_float(self):
        """Test __repr__ with float."""
        t = Tensor(3.14)
        assert repr(t) == "Tensor(3.14)"

    def test_repr_zero(self):
        """Test __repr__ with zero."""
        t = Tensor(0)
        assert repr(t) == "Tensor(0)"

    def test_repr_negative(self):
        """Test __repr__ with negative value."""
        t = Tensor(-5)
        assert repr(t) == "Tensor(-5)"


class TestEnsureTensor:
    """Test _ensure_tensor helper method."""

    def test_ensure_tensor_with_tensor(self):
        """Test _ensure_tensor with Tensor input."""
        t = Tensor(5)
        result = t._ensure_tensor(t)
        assert result is t

    def test_ensure_tensor_with_int(self):
        """Test _ensure_tensor with integer input."""
        t = Tensor(5)
        result = t._ensure_tensor(10)
        assert isinstance(result, Tensor)
        assert result.data == 10

    def test_ensure_tensor_with_float(self):
        """Test _ensure_tensor with float input."""
        t = Tensor(5)
        result = t._ensure_tensor(2.5)
        assert isinstance(result, Tensor)
        assert result.data == 2.5


class TestAddition:
    """Test addition operations."""

    def test_add_two_tensors(self):
        """Test addition of two Tensors."""
        t1 = Tensor(5)
        t2 = Tensor(3)
        result = t1 + t2
        assert result.data == 8
        assert result.op == '+'
        assert result.parents == [t1, t2]

    def test_add_tensor_and_int(self):
        """Test addition of Tensor and integer."""
        t = Tensor(5)
        result = t + 3
        assert result.data == 8

    def test_add_tensor_and_float(self):
        """Test addition of Tensor and float."""
        t = Tensor(5)
        result = t + 2.5
        assert result.data == 7.5

    def test_radd_int_and_tensor(self):
        """Test reverse addition (int + Tensor)."""
        t = Tensor(5)
        result = 3 + t
        assert result.data == 8

    def test_radd_float_and_tensor(self):
        """Test reverse addition (float + Tensor)."""
        t = Tensor(5)
        result = 2.5 + t
        assert result.data == 7.5

    def test_add_with_zero(self):
        """Test addition with zero."""
        t = Tensor(5)
        result = t + 0
        assert result.data == 5

    def test_add_negative(self):
        """Test addition with negative value."""
        t = Tensor(5)
        result = t + (-3)
        assert result.data == 2


class TestSubtraction:
    """Test subtraction operations."""

    def test_sub_two_tensors(self):
        """Test subtraction of two Tensors."""
        t1 = Tensor(10)
        t2 = Tensor(3)
        result = t1 - t2
        assert result.data == 7
        assert result.op == '-'
        assert result.parents == [t1, t2]

    def test_sub_tensor_and_int(self):
        """Test subtraction of Tensor and integer."""
        t = Tensor(10)
        result = t - 3
        assert result.data == 7

    def test_sub_tensor_and_float(self):
        """Test subtraction of Tensor and float."""
        t = Tensor(10)
        result = t - 2.5
        assert result.data == 7.5

    def test_rsub_int_and_tensor(self):
        """Test reverse subtraction (int - Tensor)."""
        t = Tensor(3)
        result = 10 - t
        assert result.data == 7

    def test_rsub_float_and_tensor(self):
        """Test reverse subtraction (float - Tensor)."""
        t = Tensor(2.5)
        result = 10 - t
        assert result.data == 7.5

    def test_sub_with_zero(self):
        """Test subtraction with zero."""
        t = Tensor(5)
        result = t - 0
        assert result.data == 5

    def test_sub_resulting_in_negative(self):
        """Test subtraction resulting in negative."""
        t = Tensor(3)
        result = t - 10
        assert result.data == -7


class TestMultiplication:
    """Test multiplication operations."""

    def test_mul_two_tensors(self):
        """Test multiplication of two Tensors."""
        t1 = Tensor(5)
        t2 = Tensor(3)
        result = t1 * t2
        assert result.data == 15
        assert result.op == '*'
        assert result.parents == [t1, t2]

    def test_mul_tensor_and_int(self):
        """Test multiplication of Tensor and integer."""
        t = Tensor(5)
        result = t * 3
        assert result.data == 15

    def test_mul_tensor_and_float(self):
        """Test multiplication of Tensor and float."""
        t = Tensor(5)
        result = t * 2.5
        assert result.data == 12.5

    def test_rmul_int_and_tensor(self):
        """Test reverse multiplication (int * Tensor)."""
        t = Tensor(5)
        result = 3 * t
        assert result.data == 15

    def test_rmul_float_and_tensor(self):
        """Test reverse multiplication (float * Tensor)."""
        t = Tensor(5)
        result = 2.5 * t
        assert result.data == 12.5

    def test_mul_by_zero(self):
        """Test multiplication by zero."""
        t = Tensor(5)
        result = t * 0
        assert result.data == 0

    def test_mul_by_negative(self):
        """Test multiplication by negative."""
        t = Tensor(5)
        result = t * (-2)
        assert result.data == -10


class TestDivision:
    """Test division operations."""

    def test_truediv_two_tensors(self):
        """Test division of two Tensors."""
        t1 = Tensor(10)
        t2 = Tensor(2)
        result = t1 / t2
        assert result.data == 5.0
        assert result.op == '/'
        assert result.parents == [t1, t2]

    def test_truediv_tensor_and_int(self):
        """Test division of Tensor and integer."""
        t = Tensor(10)
        result = t / 2
        assert result.data == 5.0

    def test_truediv_tensor_and_float(self):
        """Test division of Tensor and float."""
        t = Tensor(10)
        result = t / 2.5
        assert result.data == 4.0

    def test_rtruediv_int_and_tensor(self):
        """Test reverse division (int / Tensor)."""
        t = Tensor(2)
        result = 10 / t
        assert result.data == 5.0

    def test_rtruediv_float_and_tensor(self):
        """Test reverse division (float / Tensor)."""
        t = Tensor(2.5)
        result = 10 / t
        assert result.data == 4.0

    def test_div_by_one(self):
        """Test division by one."""
        t = Tensor(5)
        result = t / 1
        assert result.data == 5.0

    def test_div_by_negative(self):
        """Test division by negative."""
        t = Tensor(10)
        result = t / (-2)
        assert result.data == -5.0

    def test_div_zero_by_number(self):
        """Test division of zero by number."""
        t = Tensor(0)
        result = t / 5
        assert result.data == 0.0

    def test_div_by_zero_raises_error(self):
        """Test division by zero raises error."""
        t = Tensor(10)
        with pytest.raises(ZeroDivisionError):
            t / 0


class TestPower:
    """Test power operations."""

    def test_pow_two_tensors(self):
        """Test power of two Tensors."""
        t1 = Tensor(2)
        t2 = Tensor(3)
        result = t1 ** t2
        assert result.data == 8
        assert result.op == 'pow'
        assert result.parents == [t1, t2]

    def test_pow_tensor_and_int(self):
        """Test power of Tensor and integer."""
        t = Tensor(2)
        result = t ** 3
        assert result.data == 8

    def test_pow_tensor_and_float(self):
        """Test power of Tensor and float."""
        t = Tensor(4)
        result = t ** 0.5
        assert result.data == 2.0

    def test_pow_to_zero(self):
        """Test power to zero."""
        t = Tensor(5)
        result = t ** 0
        assert result.data == 1

    def test_pow_to_one(self):
        """Test power to one."""
        t = Tensor(5)
        result = t ** 1
        assert result.data == 5

    def test_pow_negative_base(self):
        """Test power with negative base."""
        t = Tensor(-2)
        result = t ** 2
        assert result.data == 4

    def test_pow_negative_exponent(self):
        """Test power with negative exponent."""
        t = Tensor(2)
        result = t ** (-1)
        assert result.data == 0.5


class TestNegation:
    """Test negation operation."""

    def test_neg_positive(self):
        """Test negation of positive Tensor."""
        t = Tensor(5)
        result = -t
        assert result.data == -5
        assert result.op == 'neg'
        assert result.parents == [t]

    def test_neg_negative(self):
        """Test negation of negative Tensor."""
        t = Tensor(-5)
        result = -t
        assert result.data == 5

    def test_neg_zero(self):
        """Test negation of zero."""
        t = Tensor(0)
        result = -t
        assert result.data == 0

    def test_double_negation(self):
        """Test double negation."""
        t = Tensor(5)
        result = -(-t)
        assert result.data == 5


class TestComplexOperations:
    """Test complex combinations of operations."""

    def test_chained_addition(self):
        """Test chained addition."""
        t1 = Tensor(5)
        t2 = Tensor(3)
        t3 = Tensor(2)
        result = t1 + t2 + t3
        assert result.data == 10

    def test_mixed_operations(self):
        """Test mixed arithmetic operations."""
        t1 = Tensor(10)
        t2 = Tensor(2)
        result = (t1 + 5) * t2
        assert result.data == 30

    def test_mixed_with_reverse_ops(self):
        """Test mixed operations with reverse operations."""
        t = Tensor(5)
        result = 10 / t - 1
        assert result.data == 1.0

    def test_complex_expression(self):
        """Test complex expression."""
        t1 = Tensor(2)
        t2 = Tensor(3)
        result = (t1 ** 2) + (t2 * 4) - 5
        assert result.data == 11

    def test_many_parents(self):
        """Test operation with result tracking."""
        t1 = Tensor(5)
        t2 = Tensor(3)
        result = t1 + t2
        assert len(result.parents) == 2
        assert result.parents[0].data == 5
        assert result.parents[1].data == 3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_number(self):
        """Test with very large numbers."""
        t = Tensor(1e100)
        result = t + 1
        assert result.data == 1e100 + 1

    def test_very_small_number(self):
        """Test with very small numbers."""
        t = Tensor(1e-100)
        result = t + 1
        assert result.data == 1 + 1e-100

    def test_scientific_notation(self):
        """Test with scientific notation."""
        t = Tensor(1.5e10)
        assert t.data == 1.5e10

    def test_tensor_with_nan_like_operations(self):
        """Test operations that result in very small differences."""
        t1 = Tensor(0.1)
        t2 = Tensor(0.2)
        t3 = Tensor(0.3)
        result = t1 + t2 - t3
        assert abs(result.data) < 1e-10


class TestImmutability:
    """Test that operations create new tensors."""

    def test_addition_creates_new_tensor(self):
        """Test that addition creates a new tensor."""
        t1 = Tensor(5)
        t2 = Tensor(3)
        result = t1 + t2
        assert result is not t1
        assert result is not t2

    def test_original_unchanged_after_add(self):
        """Test that original tensor is unchanged after addition."""
        t = Tensor(5)
        original_data = t.data
        _ = t + 3
        assert t.data == original_data

    def test_original_unchanged_after_mul(self):
        """Test that original tensor is unchanged after multiplication."""
        t = Tensor(5)
        original_data = t.data
        _ = t * 2
        assert t.data == original_data

    def test_negation_creates_new_tensor(self):
        """Test that negation creates a new tensor."""
        t = Tensor(5)
        result = -t
        assert result is not t
        assert t.data == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
