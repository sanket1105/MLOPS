from src.math_operations import add, sub


def test_add():
    assert add(2, 5) == 7
    assert add(2, -2) == 0
    assert add(2, 0) == 2
    assert add(5, 5) == 10


def test_sub():
    assert sub(1, 2) == -1
    assert sub(1, -1) == 2
