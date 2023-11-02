import numpy as np
import numexpr as ne

class memoized_expr:
    ''' Deferred computation of elementwise operations. '''

    def __init__(self, expr: str, variables: dict, shape: list):
        self.expr = expr
        self.local_dict = variables
        self.shape = shape 

    def evaluate(self) -> np.array:
        out = np.empty(self.shape, dtype=np.float32, order='F')
        ne.evaluate(self.expr, self.local_dict, casting='unsafe', out=out)

        return out