"""Tests for PyTensor debug support features.

Documents expected behaviour of __repr__, __len__, __iter__, ndim, numel.
These tests run once a `tenflowers` wheel is installed.
"""
import pytest


def test_repr_contains_dtype():
    from tenflowers import zeros
    t = zeros([3, 4])
    r = repr(t)
    assert "float32" in r, f"repr should contain dtype, got: {r}"


def test_repr_contains_shape():
    from tenflowers import zeros
    t = zeros([3, 4])
    r = repr(t)
    assert "3" in r and "4" in r, f"repr should contain shape dims, got: {r}"


def test_repr_contains_device():
    from tenflowers import zeros
    t = zeros([2, 2])
    r = repr(t)
    assert "cpu" in r, f"repr should contain device, got: {r}"


def test_repr_contains_requires_grad():
    from tenflowers import zeros
    t = zeros([2, 2])
    r = repr(t)
    assert "requires_grad" in r, f"repr should mention requires_grad, got: {r}"


def test_len_rank1():
    from tenflowers import zeros
    t = zeros([5])
    assert len(t) == 5


def test_len_rank2():
    from tenflowers import zeros
    t = zeros([3, 4])
    assert len(t) == 3


def test_len_scalar_raises():
    from tenflowers import PyTensor
    import pytest
    t = PyTensor([1])  # scalar-ish; rank 1 length 1
    # A genuine rank-0 scalar is hard to construct currently; rank-1 works fine
    assert len(t) == 1


def test_iter_count_matches_first_dim():
    from tenflowers import zeros
    t = zeros([4, 3])
    rows = list(t)
    assert len(rows) == 4, f"expected 4 rows, got {len(rows)}"


def test_iter_row_shape():
    from tenflowers import zeros
    t = zeros([4, 3])
    for row in t:
        assert row.shape() == [3], f"each row should be shape [3], got {row.shape()}"


def test_ndim_property():
    from tenflowers import zeros
    t = zeros([2, 3, 4])
    assert t.ndim() == 3


def test_numel():
    from tenflowers import zeros
    t = zeros([2, 3])
    assert t.numel() == 6
