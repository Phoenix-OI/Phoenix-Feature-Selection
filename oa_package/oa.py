import numpy as np

class OrthogonalArray:
    """The orthogonal array (OA) of iga  
    
    Parameters
    ----------
        f : int
           Factors of OA.

    Attributes
    ----------
        rows : int
            Required row by the factors of OA.
        
        data :
            The array itself of OA.
        
    """
    def __init__(self, f):
        self.factors = f
        self.rows = self.RequiredRows(f)
        self.data = np.zeros((self.rows, self.factors))

        for i in range(0, self.rows):
            for j in range(0, self.factors):
                self.data[i][j] = self.Imm(i, j, self.factors)

    def RequiredRows(self, f):
        row = 1
        f = int(f)
        while f:
            f >>= 1
            row <<= 1

        return row

    def Imm(self, row, col, factor):
        value = 0
        mask = 1

        factor >>= 1
        while factor:
            mask = mask << 1
            factor >>= 1

        col += 1
        while col:
            if col & 1:
                if (row & mask):
                    tmp = 1
                else:
                    tmp = 0

                value ^= tmp
            col >>= 1
            mask >>= 1

        return value
