from numpy import ceil, floor, pi

DEFAULT_THRESHOLD = 0.000001


def force_between_0_and_2pi(a:float) -> float:
    while a<0:
        a += 2*pi
    while a>=2*pi:
        a -= 2*pi
    return a


def force_zero_for_small_numbers(x:float|int, threshold:float=DEFAULT_THRESHOLD) -> float|int:
    if abs(x)<threshold:
        return 0
    else:
        return x
    

def force_integers_on_exactly_round(x:float|int) -> float|int:
    int_x = int(round(x))
    if int_x==x:
        return int_x
    return x


def force_integers_on_close_to_round(x:float|int, threshold:float=DEFAULT_THRESHOLD) -> float|int:
    int_x = int(round(x))
    if abs(int_x-x)<threshold:
        return int_x
    return x


def furthest_absolute_integer(x:float|int) -> int:
    if x<0:
        return int(floor(x))
    else:
        return int(ceil(x))

def force_near_pure_complex(x:complex, threshold:float=DEFAULT_THRESHOLD) -> complex|float|int:
    if abs(x.real)<threshold:
        x = 1j*force_integers_on_close_to_round(x.imag)
    if abs(x.imag)<threshold:
        x = force_integers_on_close_to_round(x.real)
    return x