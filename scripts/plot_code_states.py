import numpy as np
from mpmath import mp

if __name__ == "__main__":
    from __init__ import add_root_to_path
    add_root_to_path()



from src.quantum.states.bosonic.codes.squid import squid_state, squid_code
from src.quantum.states.bosonic.codes.cat import cat_code
from src.quantum.states.fock import FockSum
from qutip import Qobj, basis, coherent, squeeze, qzero

from src.quantum.visualizations.wigner_function import plot_plain_wigner, qutip_wigner_plot

from src.utils.visuals import matplotlib_support, plt
from src.utils.visuals.matplotlib_support import Axes

from src.quantum.qutip_support._common import print_fock

from projects.controlled_squeezing.src.visualizations import plot_light_states, plot_fock_distribution
from projects.controlled_squeezing.src.squeezing_direction import squeezing_direction_to_squeezing_phase
from projects.controlled_squeezing.src.codes_built_in_superposition import simple_m_legged_code



from typing import TypeAlias, Literal, Generator, cast
_CodeTypes : TypeAlias = Literal["cat", "squeeze"]
_QutipOrMineLiteral : TypeAlias = Literal["qutip", "mine"]
_CodeCreationMethod : TypeAlias = Literal["qutip", "my classes", "simple"]



def _iterate_states(code:list[Qobj], axes, num_digits:int, num_rows:int) -> Generator[tuple[int, Qobj, str, Axes], None, None]:
    k = -1

    if num_rows == 1:
        for i, ax in enumerate(axes):
            k += 1
            state = code[k]
            s = f"{i}"
            yield i, state, s, ax

    else:
        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                k += 1
                state = code[k]

                if i==0:
                    s = f"{j}"
                elif i:
                    if j==0:
                        s = "+"
                    else:
                        s = "-"

                yield i, state, s, ax


def plot_codes(
    num_legs:int,
    only_0:bool = False,
    plot_method:_QutipOrMineLiteral|Literal["off"] = "mine",
    code_creation_method:_CodeCreationMethod = "simple",
    code_type:_CodeTypes = "squeeze",
    plot_x_states:bool = False,
    minimalist:bool = True,
    high_res:bool = False,
    transparent_zero:bool = True,
    colorlims = None #(-0.26, +0.26)
) -> None:
    
    num_digits = 1 if only_0 else 2
    num_rows = 2 if plot_x_states else 1
    base_fig_size = 2.7


    if plot_method == "off":
        _with_plot = False
        axs = [None]*(num_rows * num_digits)
    else:
        _with_plot = True
        args = [num_rows, num_digits]
        kwargs = dict(
            figsize=(base_fig_size*num_digits, base_fig_size*num_rows)
        )
        if high_res:
            kwargs['dpi'] = 500  # type: ignore
        fig, axs = plt.subplots(*args, **kwargs)

    if num_digits == 1:
        axs = [axs]

    if high_res:
        num_points = 500
        num_moments = 500
    else:
        num_points = 200
        num_moments = 100

    print(f"\n{code_type} code with {num_legs} legs:")

    ## Set super title:
    super_title = f"{num_legs}-legged {code_type} code" 
    if code_creation_method == "qutip":
        method_str = f"Using qutip"
    elif code_creation_method == "my classes":
        method_str = f"Approximation"
    elif code_creation_method == "simple":
        method_str = f"Simple code"
    # super_title += "\n" + method_str
    if not minimalist and _with_plot:
        fig.suptitle(super_title)

    ## Create the squid code and plot:
    match code_type:
        case "cat":     r:float = 3.0
        case "squeeze": r:float = 1.5
        case _: raise ValueError(f"Unknown code type: {code_type!r}")
    

    if code_creation_method == "simple":
        if num_legs == 1:
            code = [
                squeeze(num_moments, r * squeezing_direction_to_squeezing_phase(θ, as_exponent=True)) @ basis(num_moments, 0)
                for θ in [0, np.pi/2]
            ]
        else:    
            code_states = simple_m_legged_code(num_legs, r, num_moments, code_type, num_digits=2)
            code = list(code_states)
    else:
        if code_type == "cat":
            code = cat_code(amp=r, num_legs=num_legs)  
        elif code_type == "squeeze":
            code = squid_code(amp=r, num_legs=num_legs)
        else:
            raise ValueError(f"Unknown code type: {code_type}")


    if plot_x_states:
        plus = (code[0] + code[1])/np.sqrt(2)
        minus = (code[0] - code[1])/np.sqrt(2)
        if isinstance(plus, FockSum) and isinstance(minus, FockSum):
            plus_norm = plus.norm
            minus_norm = minus.norm
        elif isinstance(plus, Qobj) and isinstance(minus, Qobj):
            plus_norm = plus.norm()
            minus_norm = minus.norm()
        else:
            raise TypeError(f"Unexpected types: {type(plus)!r}, {type(minus)!r}")
        
        assert np.isclose(plus_norm, 1.0), "Plus state is not normalized."
        assert np.isclose(minus_norm, 1.0), "Minus state is not normalized."
        code += [plus, minus]


    for i, state, state_name, ax in _iterate_states(code, axs, num_digits, num_rows):

        if only_0 and i>0:
            break

        if code_creation_method == "qutip":
            assert isinstance(state, FockSum) 
            q : Qobj = state.to_qutip(num_moments=num_moments)
            state_for_wigner = q
        elif code_creation_method == "mine":
            # f = state.to_fock(num_moments=num_moments, estimation="approximation")
            state_for_wigner = state
        elif code_creation_method == "simple":
            assert isinstance(state, Qobj) 
            state_for_wigner = state

        
        if plot_method == "mine":
            d = plot_plain_wigner(state_for_wigner, ax=ax, title=False, colorlims=colorlims, num_points=num_points, transparent_zero=transparent_zero)
        elif plot_method == "qutip":
            d = qutip_wigner_plot(q, ax=ax)
        elif plot_method == "off":
            pass
        else:
            raise ValueError(f"Unknown plot method: {plot_method}")
        
        print_fock(state, f"    |{state_name}>_{num_legs}:  ", max_terms=10)
        # plot_fock_distribution(state, title=f"|{state_name}>_{num_legs}")

        if not minimalist and _with_plot:
            title = r"$\left|{s}\right\rangle$".format(s=state_name)
            d['ax'].set_title(title)

        ## If minimalist, remove grid, ticks, labels and box:
        if minimalist and _with_plot:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            # remove box:
            for side in ['right', 'left', 'top', 'bottom']:
                ax.spines[side].set_visible(False)
            # Remove colorbar:
            if d.get('cb') is not None:
                d['cb'].remove()
            # Transparent background:
            ax.set_facecolor("none")



    ## adjust and save: 
    if _with_plot:
        fig.tight_layout()
        matplotlib_support.save_figure(fig, f"d={num_digits} {num_legs}-legged {code_type}-code - {method_str}", dpi=500, transparent=True)

    print("Done.")
    


def plot_competing_codes(
):
    n : int
    code_type : _CodeTypes
    def _call():
        plot_codes(num_legs=n, code_type=code_type, plot_method="mine", plot_x_states=False, minimalist=True)
        

    n = 2
    code_type = "cat"    
    _call()

    n = 4
    code_type = "cat"
    _call()
    
    n = 2
    code_type = "squeeze"
    _call()

    matplotlib_support.save_all_figures(["png"])

    print("Done.")


def plot_all_codes(
    n_vec = [1, 2, 4, 8],
    plot_on:bool = True,
    only_0:bool = False,
    high_res:bool = True,
):
    plot_method = "mine" if plot_on else "off"

    for n in n_vec:
        print(f"n={n}:")
        # for code_type in _CodeTypes.__args__:
        code_type = "squeeze"
        if True:
            plot_codes(num_legs=n, code_type=code_type, plot_method=plot_method, only_0=only_0, high_res=high_res)
            print(f"")

    print("Done.")



def main():
    plot_all_codes()
    # plot_competing_codes()
    # plot_codes(num_legs=6, code_type="squeeze", plot_method="mine")

if __name__ == "__main__":
    main()