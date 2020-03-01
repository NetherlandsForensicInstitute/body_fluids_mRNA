from rna.lr_system import get_mixture_columns_for_class


def test_get_mixture_columns_for_class():
    N = 8

    target_class = [1, 0, 0, 0, 0, 0, 0, 0]
    assert get_mixture_columns_for_class(target_class) == [i*2 for i in range(2**(N-1))]
    target_class = [0, 0, 0, 0, 0, 0, 0, 1]
    assert get_mixture_columns_for_class(target_class) == [i for i in range(2**(N-1))]

    # NB empty class has no single cell type in it!
    target_class = [1] * N
    assert get_mixture_columns_for_class(target_class) == [i for i in range(2**N) if i <2**N - 1]



