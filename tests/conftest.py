"""Pytest configuration and fixtures for micro_torch tests."""

import pytest
from micro_torch import Tensor


@pytest.fixture
def tensor_5():
    """Fixture providing a Tensor with value 5."""
    return Tensor(5)


@pytest.fixture
def tensor_10():
    """Fixture providing a Tensor with value 10."""
    return Tensor(10)


@pytest.fixture
def tensor_2_5():
    """Fixture providing a Tensor with value 2.5."""
    return Tensor(2.5)


@pytest.fixture
def tensor_0():
    """Fixture providing a Tensor with value 0."""
    return Tensor(0)
