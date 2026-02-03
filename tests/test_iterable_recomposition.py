from itertools import chain

import pytest

from lm_pub_quiz.util import ReversibleChain, chain_with_sizes, unchain

TEST_INPUTS = [[[1], [2, 3], [4, 5, 6]], [[], [1], []], [[]], [[1], [2]], [[], [], [1]]]


@pytest.mark.parametrize("inputs", TEST_INPUTS)
def test_decomposition_consum_chain_first(inputs):
    sizes, joined = chain_with_sizes(iter(inputs))
    assert len(list(joined)) == sum(sizes)

    with pytest.raises(StopIteration):
        next(sizes)

    with pytest.raises(StopIteration):
        next(joined)


@pytest.mark.parametrize("inputs", TEST_INPUTS)
def test_decomposition_consum_sizes_first(inputs):
    sizes, joined = chain_with_sizes(iter(inputs))
    assert sum(sizes) == len(list(joined))

    with pytest.raises(StopIteration):
        next(sizes)

    with pytest.raises(StopIteration):
        next(joined)


@pytest.mark.parametrize("inputs", TEST_INPUTS)
def test_decomposition_chain_content(inputs):
    sizes, joined = chain_with_sizes(iter(inputs))

    for size, item in zip(sizes, inputs):
        assert len(item) == size

    assert list(joined) == list(chain(*inputs))


@pytest.mark.parametrize("inputs", TEST_INPUTS)
def test_recomposition(inputs):
    sizes, joined = chain_with_sizes(iter(inputs))

    result = list(unchain(joined, sizes=sizes))

    assert result == inputs


@pytest.mark.parametrize(
    "a,b",
    [
        ([[1], [2], [3]], [[1], [2], [3, 4]]),  # Different length in last sequence
        ([[1], [2]], [[1], [2], [3]]),  # Different number of sequences
        ([[], [1], []], [[1], [], []]),  # With empty sequences
    ],
)
def test_chain_different_sizes(a, b):
    chained = ReversibleChain({"a": a, "b": b})

    with pytest.raises(ValueError):
        for _ in chained.sizes:
            pass


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([[1, 2], [3]], [[11, 22], [33]], [[12, 24], [36]]),
        ([[], [1, 2], [3]], [[], [11, 22], [33]], [[], [12, 24], [36]]),
    ],
)
def test_chain_sum(a, b, expected):
    chained = ReversibleChain({"a": a, "b": b})

    result = (a + b for a, b in zip(chained["a"], chained["b"]))

    for e, r in zip(expected, chained.reverse(result)):
        assert e == r


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([[1, 2], [3]], [[11, 22], [33]], [[12, 24], [36]]),
        ([[], [1, 2], [3]], [[], [11, 22], [33]], [[], [12, 24], [36]]),
    ],
)
def test_chain_iter_zipped_loop(a, b, expected):
    # Make sure we only iter over a and b once
    a, b = iter(a), iter(b)

    # Apply the chain
    chained = ReversibleChain({"a": a, "b": b})

    # Setup the pipeline: Within the loop, all elements are chained
    result = (a + b for a, b in zip(chained["a"], chained["b"]))

    # Consume the results
    for e, (r, inputs) in zip(expected, chained.reverse(result, yield_inputs=True)):
        r_direct = [a_ + b_ for a_, b_ in zip(inputs["a"], inputs["b"])]
        assert e == r_direct
        assert e == r
