from buggy import running_max


def test_basic():
    assert running_max([1, 2, 1, 3]) == [1, 2, 2, 3]


def test_negatives():
    assert running_max([-3, -5, -2]) == [-3, -3, -2]


def test_single():
    assert running_max([7]) == [7]
