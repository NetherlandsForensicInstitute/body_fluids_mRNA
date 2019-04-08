from rna.analytics import get_mixture_columns_for_class
from rna.constants import single_cell_types


def test_get_mixture_columns_for_class():
    # without priors
    N = len(single_cell_types)
    priors = [0.5]*N

    target_class = [1, 0, 0, 0, 0, 0, 0, 0]
    assert get_mixture_columns_for_class(target_class, priors) == [i+2**7 for i in range(2**(N-1))]
    target_class = [0, 0, 0, 0, 0, 0, 0, 1]
    assert get_mixture_columns_for_class(target_class, priors) == [i*2+1 for i in range(2**(N-1))]

    # NB empty class has no single cell type in it!
    target_class = [1] * len(single_cell_types)
    assert get_mixture_columns_for_class(target_class, priors) == [i for i in range(2**N) if i > 0]

    # with priors
    priors[1] = 1

    target_class = [0]*N
    target_class[0] = 1
    assert get_mixture_columns_for_class(target_class, priors) == [i+2**(N-1)+2**(N-2) for i in range(2**(N-2))]

    priors = [0.5]*len(single_cell_types)
    priors[0] = 1
    target_class = [0]*N
    target_class[-1] = 1
    assert get_mixture_columns_for_class(target_class, priors) == [i*2+1+2**(N-1) for i in range(2**(N-2))]

    # no possibilities
    priors = [0.5]*len(single_cell_types)
    priors[0] = 0
    target_class = [0]*N
    target_class[0] = 1
    assert get_mixture_columns_for_class(target_class, priors) == []


