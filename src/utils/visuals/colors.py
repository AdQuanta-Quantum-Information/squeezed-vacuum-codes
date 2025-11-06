from typing import Generator, Literal, overload, Tuple, TypeAlias

## For OOP:
from enum import Enum

# Color transformations 
import colorsys

_RgbFloatTuple: TypeAlias = Tuple[float, float, float]
_RgbIntTuple: TypeAlias = Tuple[int, int, int]
_AcceptedColorInput: TypeAlias = str | _RgbIntTuple | _RgbFloatTuple


@overload
def hsv_to_rgb(h: float, s: float, v: float, in_range_1_0: Literal[False]) -> Tuple[int, int, int]: ...
@overload
def hsv_to_rgb(h: float, s: float, v: float, in_range_1_0: Literal[True]) -> Tuple[float, float, float]: ...
@overload
def hsv_to_rgb(h: float, s: float, v: float, in_range_1_0: bool = True) -> Tuple[float, float, float]: ...
def hsv_to_rgb(h: float, s: float, v: float, in_range_1_0: Literal[True, False] = True) -> Tuple[float, float, float]:
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
    if in_range_1_0:
        return (r, g, b) 
    else:
        return (int(255*r), int(255*g), int(255*b)) 
 

def distinct_colors(n:int) -> Generator[tuple[float, float, float], None, None]: 
    hue_fraction = 1.0 / (n + 1) 
    return (hsv_to_rgb(hue_fraction * i, 1.0, 1.0) for i in range(0, n)) 


def color_gradient(num_colors:int):
    for i in range(num_colors):
        rgb = colorsys.hsv_to_rgb(i / num_colors, 1.0, 1.0)
        yield rgb


class ColorsByName(Enum):
    aliceblue            = '#F0F8FF'
    antiquewhite         = '#FAEBD7'
    aqua                 = '#00FFFF'
    aquamarine           = '#7FFFD4'
    azure                = '#F0FFFF'
    beige                = '#F5F5DC'
    bisque               = '#FFE4C4'
    black                = '#000000'
    blanchedalmond       = '#FFEBCD'
    blue                 = '#0000FF'
    blueviolet           = '#8A2BE2'
    brown                = '#A52A2A'
    burlywood            = '#DEB887'
    cadetblue            = '#5F9EA0'
    chartreuse           = '#7FFF00'
    chocolate            = '#D2691E'
    coral                = '#FF7F50'
    cornflowerblue       = '#6495ED'
    cornsilk             = '#FFF8DC'
    crimson              = '#DC143C'
    cyan                 = '#00FFFF'
    darkblue             = '#00008B'
    darkcyan             = '#008B8B'
    darkgoldenrod        = '#B8860B'
    darkgray             = '#A9A9A9'
    darkgreen            = '#006400'
    darkkhaki            = '#BDB76B'
    darkmagenta          = '#8B008B'
    darkolivegreen       = '#556B2F'
    darkorange           = '#FF8C00'
    darkorchid           = '#9932CC'
    darkred              = '#8B0000'
    darksalmon           = '#E9967A'
    darkseagreen         = '#8FBC8F'
    darkslateblue        = '#483D8B'
    darkslategray        = '#2F4F4F'
    darkturquoise        = '#00CED1'
    darkviolet           = '#9400D3'
    deeppink             = '#FF1493'
    deepskyblue          = '#00BFFF'
    dimgray              = '#696969'
    dodgerblue           = '#1E90FF'
    firebrick            = '#B22222'
    floralwhite          = '#FFFAF0'
    forestgreen          = '#228B22'
    fuchsia              = '#FF00FF'
    gainsboro            = '#DCDCDC'
    ghostwhite           = '#F8F8FF'
    gold                 = '#FFD700'
    goldenrod            = '#DAA520'
    gray                 = '#808080'
    green                = '#008000'
    greenyellow          = '#ADFF2F'
    honeydew             = '#F0FFF0'
    hotpink              = '#FF69B4'
    indianred            = '#CD5C5C'
    indigo               = '#4B0082'
    ivory                = '#FFFFF0'
    khaki                = '#F0E68C'
    lavender             = '#E6E6FA'
    lavenderblush        = '#FFF0F5'
    lawngreen            = '#7CFC00'
    lemonchiffon         = '#FFFACD'
    lightblue            = '#ADD8E6'
    lightcoral           = '#F08080'
    lightcyan            = '#E0FFFF'
    lightgoldenrodyellow = '#FAFAD2'
    lightgreen           = '#90EE90'
    lightgray            = '#D3D3D3'
    lightpink            = '#FFB6C1'
    lightsalmon          = '#FFA07A'
    lightseagreen        = '#20B2AA'
    lightskyblue         = '#87CEFA'
    lightslategray       = '#778899'
    lightsteelblue       = '#B0C4DE'
    lightyellow          = '#FFFFE0'
    lime                 = '#00FF00'
    limegreen            = '#32CD32'
    linen                = '#FAF0E6'
    magenta              = '#FF00FF'
    maroon               = '#800000'
    mediumaquamarine     = '#66CDAA'
    mediumblue           = '#0000CD'
    mediumorchid         = '#BA55D3'
    mediumpurple         = '#9370DB'
    mediumseagreen       = '#3CB371'
    mediumslateblue      = '#7B68EE'
    mediumspringgreen    = '#00FA9A'
    mediumturquoise      = '#48D1CC'
    mediumvioletred      = '#C71585'
    midnightblue         = '#191970'
    mintcream            = '#F5FFFA'
    mistyrose            = '#FFE4E1'
    moccasin             = '#FFE4B5'
    navajowhite          = '#FFDEAD'
    navy                 = '#000080'
    oldlace              = '#FDF5E6'
    olive                = '#808000'
    olivedrab            = '#6B8E23'
    orange               = '#FFA500'
    orangered            = '#FF4500'
    orchid               = '#DA70D6'
    palegoldenrod        = '#EEE8AA'
    palegreen            = '#98FB98'
    paleturquoise        = '#AFEEEE'
    palevioletred        = '#DB7093'
    papayawhip           = '#FFEFD5'
    peachpuff            = '#FFDAB9'
    peru                 = '#CD853F'
    pink                 = '#FFC0CB'
    plum                 = '#DDA0DD'
    powderblue           = '#B0E0E6'
    purple               = '#800080'
    red                  = '#FF0000'
    rosybrown            = '#BC8F8F'
    royalblue            = '#4169E1'
    saddlebrown          = '#8B4513'
    salmon               = '#FA8072'
    sandybrown           = '#FAA460'
    seagreen             = '#2E8B57'
    seashell             = '#FFF5EE'
    sienna               = '#A0522D'
    silver               = '#C0C0C0'
    skyblue              = '#87CEEB'
    slateblue            = '#6A5ACD'
    slategray            = '#708090'
    snow                 = '#FFFAFA'
    springgreen          = '#00FF7F'
    steelblue            = '#4682B4'
    tan                  = '#D2B48C'
    teal                 = '#008080'
    thistle              = '#D8BFD8'
    tomato               = '#FF6347'
    turquoise            = '#40E0D0'
    violet               = '#EE82EE'
    wheat                = '#F5DEB3'
    white                = '#FFFFFF'
    whitesmoke           = '#F5F5F5'
    yellow               = '#FFFF00'
    yellowgreen          = '#9ACD32'
    ## Tab colors:
    tab_blue             = '#1f77b4'
    tab_orange           = '#ff7f0e'
    tab_green            = '#2ca02c'
    tab_red              = '#d62728'
    tab_purple           = '#9467bd'
    tab_brown            = '#8c564b'
    tab_pink             = '#e377c2'
    tab_gray             = '#7f7f7f'
    tab_olive            = '#bcbd22'
    tab_cyan             = '#17becf'


def _color_input_to_rgb_float(color: _AcceptedColorInput) -> _RgbFloatTuple:
    if isinstance(color, str):
        color = color.lower()
        if color.startswith('#') and len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return _rgb_int_to_float((r, g, b))
        else:
            try:
                color_enum = ColorsByName[color]
                hex_color = color_enum.value
                r = int(hex_color[1:3], 16)
                g = int(hex_color[3:5], 16)
                b = int(hex_color[5:7], 16)
                return _rgb_int_to_float((r, g, b))
            except KeyError:
                raise ValueError(f"Color name '{color}' not recognized.") 
    elif isinstance(color, tuple) and len(color) == 3:
        if all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            return _rgb_int_to_float(color)  #type: ignore
        elif all(isinstance(c, float) and 0.0 <= c <= 1.0 for c in color):
            return color  
        else:
            raise ValueError("RGB tuple values must be in the range 0-255 (int) or 0.0-1.0 (float).")
    else:
        raise TypeError("Color must be a hex string or an RGB tuple.")


def _rgb_int_to_float(rgb: _RgbIntTuple) -> _RgbFloatTuple:
    return (rgb[0]/255, rgb[1]/255, rgb[2]/255)


def color_shades(color: _AcceptedColorInput, num: int) -> list[_RgbFloatTuple]:
    r, g, b = _color_input_to_rgb_float(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    ## Create `num` shades by varying the value (brightness)
    shades = []
    for i in range(num):
        # Adjust the brightness (value) while keeping hue and saturation constant
        new_v = v * (1 - i / num)
        r, g, b = colorsys.hsv_to_rgb(h, s, new_v)
        shades.append(_rgb_int_to_float((int(r * 255), int(g * 255), int(b * 255))))
    return shades



def tab_color(name:str) -> str:
    """ Tab colors are the matplotlib default colors. """



## -------- tests: -------- ##

def _test():
    import matplotlib.pyplot as plt

    color = "blue"
    num = 5
    colors = color_shades(color, num)
    ax = plt.gca()
    for i, color in enumerate(colors):
        print(f"Shade {i + 1}: {color}")
        ax.plot([0, 1], [i, i], color=color, linewidth=15, label=f"{color}")
    
    plt.legend()
    print("Done.")


if __name__ == "__main__":
    _test()