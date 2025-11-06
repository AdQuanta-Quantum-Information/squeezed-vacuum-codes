"""
Binary utilities for quantum information research.

This module provides helper functions and types for working with binary representations,
particularly useful for multi-qubit computational basis states.
"""

from typing import Union, List, Generator, TypeAlias
from collections.abc import Sequence


class Binary(int):
    """
    A binary digit type that can only be 0 or 1.
    
    This is a subclass of int with validation to ensure the value is only 0 or 1.
    """
    
    def __new__(cls, value: Union[int, str, bool]) -> 'Binary':
        # Convert to int first
        if isinstance(value, str):
            int_value = int(value)
        elif isinstance(value, bool):
            int_value = int(value)
        else:
            int_value = int(value)
        
        # Validate that it's binary
        if int_value not in (0, 1):
            raise ValueError(f"Binary value must be 0 or 1, got {int_value}")
        
        return super().__new__(cls, int_value)
    
    def __repr__(self) -> str:
        return f"Binary({int(self)})"
    
    def __str__(self) -> str:
        return str(int(self))


BinarySequence: TypeAlias = List[Binary]


def binary_to_decimal(binary_sequence: Sequence[Union[Binary, int]]) -> int:
    """
    Convert a sequence of binary digits to a decimal integer.
    
    Args:
        binary_sequence: A sequence of binary digits (0s and 1s), where the first
                        element is the most significant bit.
    
    Returns:
        The decimal representation of the binary sequence.
    
    Examples:
        >>> binary_to_decimal([1, 0, 1])  # 101 in binary
        5
        >>> binary_to_decimal([0, 0, 1])  # 001 in binary
        1
        >>> binary_to_decimal([Binary(1), Binary(1), Binary(0)])  # 110 in binary
        6
    """
    decimal = 0
    for i, bit in enumerate(binary_sequence):
        # Validate bit is 0 or 1
        if int(bit) not in (0, 1):
            raise ValueError(f"All elements must be binary (0 or 1), got {bit} at position {i}")
        
        # Calculate decimal value: bit * 2^(n-1-i) where n is the length
        power = len(binary_sequence) - 1 - i
        decimal += int(bit) * (2 ** power)
    
    return decimal


def decimal_to_binary(decimal: int, num_bits: int) -> BinarySequence:
    """
    Convert a decimal integer to a sequence of binary digits.
    
    Args:
        decimal: The decimal integer to convert.
        num_bits: The number of bits to use for the representation.
    
    Returns:
        A list of Binary objects representing the binary sequence, with the most
        significant bit first.
    
    Raises:
        ValueError: If the decimal number requires more than num_bits to represent.
    
    Examples:
        >>> decimal_to_binary(5, 3)  # 5 = 101 in binary
        [Binary(1), Binary(0), Binary(1)]
        >>> decimal_to_binary(1, 3)  # 1 = 001 in binary  
        [Binary(0), Binary(0), Binary(1)]
        >>> decimal_to_binary(6, 3)  # 6 = 110 in binary
        [Binary(1), Binary(1), Binary(0)]
    """
    if decimal < 0:
        raise ValueError(f"Decimal number must be non-negative, got {decimal}")
    
    max_value = (2 ** num_bits) - 1
    if decimal > max_value:
        raise ValueError(f"Decimal {decimal} requires more than {num_bits} bits to represent (max: {max_value})")
    
    # Convert to binary string, remove '0b' prefix, and pad with zeros
    binary_str = bin(decimal)[2:].zfill(num_bits)
    
    # Convert each character to Binary object
    return [Binary(int(bit)) for bit in binary_str]


def binary_counting_generator(num_bits: int) -> Generator[BinarySequence, None, None]:
    """
    Generate all possible binary sequences of a given length in counting order.
    
    This generator yields binary sequences from 000...0 to 111...1 (for num_bits bits)
    in ascending decimal order.
    
    Args:
        num_bits: The number of bits in each binary sequence.
    
    Yields:
        Binary sequences in counting order: [0,0,0], [0,0,1], [0,1,0], [0,1,1], etc.
    
    Examples:
        >>> list(binary_counting_generator(2))
        [[Binary(0), Binary(0)], [Binary(0), Binary(1)], [Binary(1), Binary(0)], [Binary(1), Binary(1)]]
        >>> list(binary_counting_generator(3))[:4]
        [[Binary(0), Binary(0), Binary(0)], [Binary(0), Binary(0), Binary(1)], 
         [Binary(0), Binary(1), Binary(0)], [Binary(0), Binary(1), Binary(1)]]
    """
    if num_bits < 0:
        raise ValueError(f"Number of bits must be non-negative, got {num_bits}")
    
    # Generate all numbers from 0 to 2^num_bits - 1
    for decimal in range(2 ** num_bits):
        yield decimal_to_binary(decimal, num_bits)


# Convenience functions for common use cases
def binary_strings_generator(num_bits: int) -> Generator[str, None, None]:
    """
    Generate binary strings in counting order.
    
    Args:
        num_bits: The number of bits in each binary string.
    
    Yields:
        Binary strings: "000", "001", "010", "011", etc.
    
    Examples:
        >>> list(binary_strings_generator(2))
        ['00', '01', '10', '11']
    """
    for binary_seq in binary_counting_generator(num_bits):
        yield ''.join(str(bit) for bit in binary_seq)


def binary_tuples_generator(num_bits: int) -> Generator[tuple, None, None]:
    """
    Generate binary tuples in counting order.
    
    Args:
        num_bits: The number of bits in each binary tuple.
    
    Yields:
        Binary tuples: (0,0,0), (0,0,1), (0,1,0), (0,1,1), etc.
    
    Examples:
        >>> list(binary_tuples_generator(2))
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    """
    for binary_seq in binary_counting_generator(num_bits):
        yield tuple(int(bit) for bit in binary_seq)


# Example usage and tests
if __name__ == "__main__":
    # Test Binary type
    print("Testing Binary type:")
    b1 = Binary(0)
    b2 = Binary(1)
    print(f"Binary(0): {b1!r}")
    print(f"Binary(1): {b2!r}")
    
    try:
        Binary(2)  # Should raise ValueError
    except ValueError as e:
        print(f"Expected error for Binary(2): {e}")
    
    print("\nTesting binary_to_decimal:")
    print(f"[1, 0, 1] -> {binary_to_decimal([1, 0, 1])}")  # Should be 5
    print(f"[0, 0, 1] -> {binary_to_decimal([0, 0, 1])}")  # Should be 1
    print(f"[1, 1, 0] -> {binary_to_decimal([1, 1, 0])}")  # Should be 6
    
    print("\nTesting decimal_to_binary:")
    print(f"5 with 3 bits -> {decimal_to_binary(5, 3)}")
    print(f"1 with 3 bits -> {decimal_to_binary(1, 3)}")
    print(f"6 with 3 bits -> {decimal_to_binary(6, 3)}")
    
    print("\nTesting binary_counting_generator (2 bits):")
    for i, binary_seq in enumerate(binary_counting_generator(2)):
        print(f"{i}: {binary_seq}")
    
    print("\nTesting binary_counting_generator (3 bits, first 4):")
    for i, binary_seq in enumerate(binary_counting_generator(3)):
        if i >= 4:
            break
        print(f"{i}: {binary_seq}")
    
    print("\nTesting binary_strings_generator (2 bits):")
    for i, binary_str in enumerate(binary_strings_generator(2)):
        print(f"{i}: {binary_str}")
