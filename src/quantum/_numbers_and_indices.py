from typing import Literal, TypeVar, Generator, Sequence, cast
import numpy as np

from ..utils import assertions

BitType = Literal[0, 1]
BitsType = list[BitType]



T = TypeVar('T')
def _in_reverse(L: list[T]) -> Generator[T, None, None]:
    N = len(L)
    for i in range(N-1, -1, -1):
        yield L[i]

def binary2dec(bits: BitsType ) -> int:
    num = 0
    # Iterate over reversed order of bits
    for n, bit in enumerate(_in_reverse(bits)):
        num += bit*(2**n)
    return num        

def dec2binary(n: int, length: int|None = None ) -> BitsType:  
    # Transform integer to binary string:
    if length is None:
        binary_str = f"{n:0b}" 
    else:
        binary_str = f"{n:0{length}b}" 
    # Transform binary string to list:
    binary_list : BitsType = [cast(BitType, int(c)) for c in binary_str]  
    return binary_list




class BinaryIndex():
    """Index: 
        helper class for 'DensityMatrix' that makes indexing the matrix more convenient and simple.
    """
    def _set_values(self, index: int, binary: BitsType) -> None:
        self.__index = index
        self.__binary = binary

    def __init__(self, mat: np.ndarray, input: int|BitsType) -> None:
        assert mat.shape[0]==mat.shape[1]
        self.length = assertions.integer(np.log2(mat.shape[0]))
        if isinstance(input, int):
            input = assertions.index(input)
            self._set_values(index=input, binary=dec2binary(input, length=self.length))
        elif isinstance(input, list):
            input = [assertions.bit(x) for x in input]
            self._set_values(index=binary2dec(input), binary=input)

    @property
    def binary(self) -> BitsType:
        return self.__binary
    @binary.setter
    def binary(self, bits: BitsType) -> None:
        self._set_values(index=binary2dec(bits), binary=bits)
    def __getitem__(self, key: int) -> BitType:
        key = assertions.index(key)
        return self.binary[key]
    def __setitem__(self, key: int, val: BitType) -> None:
        key = assertions.index(key)
        val = assertions.bit(val)
        binary = self.__binary
        binary[key] = val
        self._set_values(index=binary2dec(binary), binary=binary)
    def __delitem__(self, key):
        raise NotImplemented
    @property
    def index(self) -> int:
        return self.__index
    @index.setter
    def index(self, ind: int):
        ind = assertions.index(ind)
        self._set_values(index=ind, binary=dec2binary(ind, length=self.length))
    def __call__(self) -> int:
        return self.index
    def __repr__(self) -> str:
        return f"{self.binary}: {self.index}"
