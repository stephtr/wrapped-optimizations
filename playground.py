#%%
import numpy as np
from src.wrapped_optimizations import differential_evolution, eval_function

def func(use_param, use_const, use_print=False):
    # define a named parameter
    x = use_param((2), 'x', bounds=(-5, 5))
    # define an unnamed parameter
    y = use_param((2,2), bounds=(-5, 5))

    # define a (named) constant
    # equals `N = 3`, but with the advantage that the value also gets saved to result.x
    N = use_const(3, name='N')

    V = use_param((N), bounds=(0, 20))
    if use_print:
        print(f'V: {V}')
    return np.linalg.norm(x - 1)**2 + np.linalg.norm(y - 3)**2

if __name__ == "__main__":
    result = differential_evolution(func, workers=-1)
    print(result)
# %%
