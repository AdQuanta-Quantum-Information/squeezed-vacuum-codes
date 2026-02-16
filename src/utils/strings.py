# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #
from . import arguments, lists, maths, numerics, types, numerics

from typing import Any, Literal, Optional, Generator

# Math:
from math import pi
import numpy as np

import time
import string 

# for basic OOP:
from enum import Enum


# ==================================================================================== #
#|                                  Constants                                         |#
# ==================================================================================== #

EPS = 1e-6

class SpecialChars:
    NewLine = "\n"
    CarriageReturn = "\r"
    Tab = "\t"
    BackSpace = "\b"    
    LineUp = '\033[1A'   # Move cursor up one line
    LineClear = '\x1b[2K'   # Clear the entire line
    FractionSlash = "\u2044"   # "⁄" symbol
    SquareRoot = "\u221A"  # "√" symbol
    Infinity = "\u221E"  # "∞" symbol
    Degree = "\u00B0"  # "°" symbol
    OuterProduct = "\u2297"  # "⊗" symbol
    InnerProduct = "\u00B7"  # "·" symbol


class GreekLetters:
    # Uppercase Greek letters
    Alpha: str = '\u0391'
    Beta: str = '\u0392'
    Gamma: str = '\u0393'
    Delta: str = '\u0394'
    Epsilon: str = '\u0395'
    Zeta: str = '\u0396'
    Eta: str = '\u0397'
    Theta: str = '\u0398'
    Iota: str = '\u0399'
    Kappa: str = '\u039A'
    Lambda: str = '\u039B'
    Mu: str = '\u039C'
    Nu: str = '\u039D'
    Xi: str = '\u039E'
    Omicron: str = '\u039F'
    Pi: str = '\u03A0'
    Rho: str = '\u03A1'
    Sigma: str = '\u03A3'
    Tau: str = '\u03A4'
    Upsilon: str = '\u03A5'
    Phi: str = '\u03A6'
    Chi: str = '\u03A7'
    Psi: str = '\u03A8'
    Omega: str = '\u03A9'

    # Lowercase Greek letters
    alpha: str = '\u03B1'
    beta: str = '\u03B2'
    gamma: str = '\u03B3'
    delta: str = '\u03B4'
    epsilon: str = '\u03B5'
    zeta: str = '\u03B6'
    eta: str = '\u03B7'
    theta: str = '\u03B8'
    iota: str = '\u03B9'
    kappa: str = '\u03BA'
    lambda_: str = '\u03BB'  # lambda is a reserved keyword in Python
    mu: str = '\u03BC'
    nu: str = '\u03BD'
    xi: str = '\u03BE'
    omicron: str = '\u03BF'
    pi: str = '\u03C0'
    rho: str = '\u03C1'
    sigma: str = '\u03C3'
    tau: str = '\u03C4'
    upsilon: str = '\u03C5'
    phi: str = '\u03C6'
    chi: str = '\u03C7'
    psi: str = '\u03C8'
    omega: str = '\u03C9'


ASCII_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ==================================================================================== #
#|                               declared classes                                     |#
# ==================================================================================== #

class StrEnum(Enum): 

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self._str_value() == other
        return super().__eq__(other)
    
    def __add__(self, other:str) -> str:
        if not isinstance(other, str):
            raise TypeError(f"other is {type(other)}")
        return self._str_value()+other
    
    def __radd__(self, other:str) -> str:
        if not isinstance(other, str):
            raise TypeError(f"other is {type(other)}")
        return other+self._str_value()
    
    def __hash__(self):
        return hash(self._str_value())
    
    def __str__(self) -> str:
        return self._str_value()
    
    def _str_value(self) -> str:
        s = self.value
        if not isinstance(s, str):
            s = self.name.lower()
        return s
        

# ==================================================================================== #
#|                              declared functions                                    |#
# ==================================================================================== #

def to_list(s:str)->list[str]:
    return [c for c in s]


ASCII_UPPERCASE_LIST = to_list(ASCII_UPPERCASE)
DIGITS_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def random(len:int=1)->str:
    s = ""
    for _ in range(len):
        s += lists.random_item(ASCII_UPPERCASE_LIST)
    return s


def random_digits(len:int=1)->str:
    s = ""
    for _ in range(len):
        s += f"{lists.random_item(DIGITS_LIST)}"
    return s


def formatted(
    val:Any, 
    fill:str=' ', 
    alignment:Literal['<','^','>']='>', 
    width:Optional[int]=None, 
    precision:Optional[int]=None,
    signed:bool=False
) -> str:
    
    # Check info:
    try:
        if round(val)==val and precision is None:
            force_int = True
        else:
            force_int = False
    except:
        force_int = False
           
    # Simple formats:
    format = f"{fill}{alignment}"
    if signed:
        format += "+"
    
    # Width:
    if isinstance(width, int):
        format += f"{width}"            
    elif width is None:
        pass
    else:
        raise TypeError(f"Unexpected type {type(width)!r}")
    
    precision = arguments.default_value(precision, 0)
    if not isinstance(val, str):
        format += f".{precision}f"    
        
    return f"{val:{format}}"  


def num_out_of_num(num1:int, num2:int) -> str:
    width = len(str(num2))
    format = lambda num: formatted(num, fill=' ', alignment='>', width=width )
    return format(num1)+"/"+format(num2)


def time_stamp() -> str:
    t = time.localtime()
    return f"{t.tm_year}.{t.tm_mon:02}.{t.tm_mday:02}_{t.tm_hour:02}.{t.tm_min:02}.{t.tm_sec:02}"


def insert_spaces_in_newlines(s:str, num_spaces:int) -> str:
    spaces = ' '*num_spaces
    s2 = s.replace('\n','\n'+spaces)
    return s2


def str_width(s:str, last_line_only:bool=False) -> int:
    lines = s.split('\n')
    widths = [len(line) for line in lines]
    if last_line_only:
        return widths[-1]
    else:
        return max(widths)


def num_lines(s:str)->int:
    n = s.count(SpecialChars.NewLine)
    return n + 1


def str_len_until_first_newline(s:str)->int:
    first_new_line = s.find(SpecialChars.NewLine)
    if first_new_line == -1:
        return len(s)
    return first_new_line


def alphabet(upper_case:bool=False)->Generator[str, None, None]:
    if upper_case is True:
        l = list( string.ascii_uppercase )
    else:
        l = list( string.ascii_lowercase )
    for s in l:
        yield s


def sqrt_or_fraction_str(x:float, eps:float=1e-5, raise_error_on_mismatch:bool=False) -> str:    
    """
    Args:
        x (float): value
        eps (float, optional): precision. Defaults to 1e-5.
        raise_error_on_mismatch (bool, optional): toggles if mismatch raises error. Defaults to True.

    Raises:
        ValueError: when the value is not a square-root of some integer

    Returns:
        str: The string representation of the square-root or fraction
    """
    ## Helper functions:
    def _closest_int(val:float) -> int:
        return int(round(val))

    def _distance_from_closest_int(val:float) -> float:
        return abs(val - _closest_int(val))
    
    if _distance_from_closest_int(x) < eps:
        return f"{_closest_int(x)}"
    
    if _distance_from_closest_int(1/x) < eps:
        s = "1"+SpecialChars.FractionSlash+f"{_closest_int(1/x)}"
        if x<0:
            s = "-"+s
        return s
    
    try:
        num, is_one_over_num, sign = maths.find_closest_int_from_sqrt_including_one_over(x, eps=eps)
    except ValueError:
        if raise_error_on_mismatch:
            raise ValueError(f"Value {x!r} is not a square-root of some integer")
        else:
            return f"{x}"
    
    if is_one_over_num:
        s = "1"+SpecialChars.FractionSlash
    else:
        s = ""
    s += SpecialChars.SquareRoot+f"{num}"

    if sign == "-":
        s = "-"+s

    return s


def simplified_angle_str(angle:float, precision:int=5) -> str:
    ## Special handling of 0:
    if abs(angle - 0) < EPS:
        return "0"
    
    ## Special handling of pi fractions:
    pi_scalar = angle / pi
    try:
        s = sqrt_or_fraction_str(pi_scalar, raise_error_on_mismatch=True)
        if s == "1":  # In case the number is 1, we don't need to show it
            s = ""
        if SpecialChars.FractionSlash in s:  # In case the number is a fraction, add $\cdot$ symbol:
            s += SpecialChars.InnerProduct
        s += SpecialChars.Pi
    except ValueError:
        s = formatted(angle, precision=precision)
    return s


def simplified_complex_str(c:complex, precision:int=5) -> str:

    ## Special handling of pure real or pure imaginary numbers:
    res = None
    if isinstance(c, int|float):  # In case weight is pure real:    
        res = _derive_simplified_real_num_str(c, precision=precision)
    elif isinstance(c, complex):
        c = numerics.force_near_pure_complex(c)
        if c.imag == 0:  # In case weight is still pure real:    
            res = _derive_simplified_real_num_str(c.real, precision=precision)  
        elif c.real == 0:  # In case weight is pure real:    
            res = _derive_simplified_real_num_str(c.imag, precision=precision, is_imaginary=True)
        else:
            pass
    else:
        raise TypeError(f"Unexpected type: {type(c)}")

    if res is not None:
        return res
    
    # Now we really have a complex number, let's deal with it in parts:

    ## Check if the real and imaginary parts are close to each other:
    if abs( abs(c.real) - abs(c.imag) ) < EPS:
        common_fraction = 1/abs(c.real)
        c *= common_fraction
        c = numerics.force_near_pure_complex(c)
    else:
        common_fraction = None

    # Deal with major part of the complex number string:
    s = "("
    s += _derive_simplified_real_num_str(c.real, precision=precision, is_imaginary=False)
    if c.imag > 0:
        s += " + "
    else:
        s += " - "
    s += _derive_simplified_real_num_str(abs(c.imag), precision=precision, is_imaginary=True)
    s += ")"

    ## In case real and imaginary parts are close to each other:
    if common_fraction is not None:
        if abs(common_fraction-1) < EPS:  # if common_fraction is 1, then we don't need to show it
            pass
        else:
            s += SpecialChars.FractionSlash + _derive_simplified_real_num_str(common_fraction, precision=precision, is_imaginary=False)
    return s


def _derive_simplified_real_num_str(x:float|int, precision:int, is_imaginary:bool=False) -> str:
    abs_x = abs(x)

    try:
        s = sqrt_or_fraction_str(abs_x, raise_error_on_mismatch=True)
    except ValueError:
        s = formatted(abs_x, precision=precision)
    
    if is_imaginary:
        if s == "1":  
            s = "j"  # In case the number is 1, we don't need to show it
        else:
            s += "j"

    if x<0:
        s = "-" + s

    return s


def mat_str(mat:np.matrix) -> str:
    # check:
    assert isinstance(mat, np.ndarray)
    m = mat.copy()
    # reduce values close to zero:
    data_type = types.numpy_dtype_to_std_type(m.dtype)
    for idx, x in np.ndenumerate(m):
        m[idx] =  numerics.force_near_pure_complex(x)
    # print:
    if np.all( np.isreal(m) ):  # if all matrix is real
        return f"{np.real(m)}"
    elif np.all( np.isreal(m*1j) ):  # if all matrix is imaginary
        return f"{np.imag(m)} * 1j"
    else:
        return f"{m}"


def mat_str_with_leading_text(mat:np.matrix, text:str) -> str:
    if mat is None:
        return text + "None"
    assert isinstance(text, str)
    text_width = str_width(text, last_line_only=True)
    s = mat_str(mat)
    return text + insert_spaces_in_newlines(s, num_spaces=text_width)




def format_float_for_as_str(value: float, use_latex: bool) -> str:
    """Format a float for use in a figure title.

    If use_latex is True, return a LaTeX math-mode string like
    "$1\times10^{-6}$" so the TeX engine renders a proper exponent.
    Otherwise return a compact plain string in scientific notation like
    "1e-06".
    """
    if value == 0:
        return r"$0$" if use_latex else "0"

    if use_latex:
        s = "{:.1e}".format(value)
        mantissa_str, exp_str = s.split("e")
        # clean mantissa (remove trailing .0)
        try:
            mant = float(mantissa_str)
        except ValueError:
            # fallback
            return rf"${s}$"
        # drop .0 when it's integer
        if mant.is_integer():
            mant_display = str(int(mant))
        else:
            mant_display = str(mant)
        exp = int(exp_str)
        # Return a LaTeX math expression
        if mant_display == "1":
            # omit the multiplicative 1 for aesthetics
            return rf"$10^{{{exp}}}$"
        return rf"${mant_display}\times10^{{{exp}}}$"
    else:
        return "{:.1e}".format(value)
