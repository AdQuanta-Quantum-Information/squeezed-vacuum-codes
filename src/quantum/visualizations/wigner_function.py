## ================== Imports ================== ##

from typing import Literal, TypedDict, Union, Optional, Final, ClassVar, TypeAlias, overload

## For plotting:
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colorbar import Colorbar
from matplotlib.contour import ContourSet
from matplotlib.transforms import Bbox
from matplotlib.colors import LightSource, TwoSlopeNorm
from matplotlib.cm import ScalarMappable


## Our utilities:
from ...utils.visuals import matplotlib_support
from ...utils.prints import ProgressBar
from ...utils import assertions


# A bit of OOP:
from dataclasses import dataclass, field

# Everyone needs numpy in their life and other math stuff:
import numpy as np
import math

from copy import deepcopy

# for quantum tools:
import qutip
from qutip import Qobj

# For wigner function on bloch sphere:
from sympy.physics.wigner import wigner_3j
from scipy.special import sph_harm
from qutip.matplotlib_utilities import complex_phase_cmap

# Tools:
import itertools

@overload
def _convert_qobj_to_np_type(m:None) -> None: ...
@overload
def _convert_qobj_to_np_type(m:np.ndarray|Qobj) -> np.ndarray: ...
def _convert_qobj_to_np_type(m:np.ndarray|Qobj|None) -> np.ndarray|None:
    if isinstance(m, Qobj):
        m = m.full()
    elif m is None:
        return None
    assert isinstance(m, np.ndarray)
    return m


class WignerOutput(TypedDict):
    fig : Figure
    ax  : Axes
    cf  : ContourSet
    cb  : Colorbar


def compute_wigner_values(
    state: np.ndarray | qutip.Qobj,
    alpha_max: float,
    num_points: int = 250,
    method: str = "clenshaw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Wigner values on a square phase-space grid.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Wigner matrix W
            - x-axis vector xvec
            - y-axis vector yvec
    """
    if isinstance(state, np.ndarray):
        rho = qutip.Qobj(state)
    elif isinstance(state, qutip.Qobj):
        rho = state
    else:
        raise TypeError(f"Invalid state type: {type(state)!r}")

    if qutip.isket(rho):
        rho = qutip.ket2dm(rho)

    xvec = np.linspace(-alpha_max, alpha_max, num_points)
    W0 = qutip.wigner(rho, xvec, xvec, method=method)
    W, yvec = W0 if isinstance(W0, tuple) else (W0, xvec)

    return np.asarray(W), np.asarray(xvec), np.asarray(yvec)


def _plot_wigner(
    rho, fig:Figure, ax:Axes,
    cmap=None, alpha_max=7.5, colorbar=False,
    colorlims:tuple[float, float]|None=None,
    method='clenshaw', projection='2d',
    num_points:int=1000,
    num_colors:int=251,
    transparent_zero: float|bool = False
)->tuple[
    Figure,         # fig, 
    Axes,       # ax, 
    ContourSet,     # cf, 
    Colorbar        # cb
]:
    """_plot_wigner Taken from qutip and slightly changed. 
    qutip are the authors of this function.
    """

    W, xvec, yvec = compute_wigner_values(
        state=rho,
        alpha_max=alpha_max,
        num_points=num_points,
        method=method,
    )

    ## Color limits:
    if colorlims is None:
        wlim = abs(W).max()
        wlims = (-wlim, wlim)
        # wlims = (W.min(), W.max())
        norm_ = mpl.colors.Normalize(wlims[0], wlims[1])

    else:
        assert len(colorlims)==2
        assert isinstance(colorlims, tuple)
        assert colorlims[0] < colorlims[1]
        _set_ndarray_within_limits(W, colorlims)
        wlims = colorlims
        norm_ = mpl.colors.TwoSlopeNorm(vmin=colorlims[0], vcenter=0.0, vmax=colorlims[1])


    if cmap is None:
        cmap = cm.get_cmap('RdBu')
    # Make a copy so we can safely tweak alpha for masked values
    try:
        cmap = cmap.copy()
    except Exception:
        cmap = deepcopy(cmap)

    # Determine masking threshold (values near zero become transparent)
    W_plot = W
    if not transparent_zero is False :
        if transparent_zero is True:
            eps = 0.05 * min((abs(val) for val in wlims))                  
        elif isinstance(transparent_zero, (int, float)):
            eps = float(transparent_zero)          
        else:
            raise TypeError(f"Invalid type of `transparent_zero`: {type(transparent_zero)!r}")
        
        W_plot = np.ma.masked_where(np.abs(W) < eps, W)
        cmap.set_bad((1, 1, 1, 0))                 # fully transparent for masked

    if projection == '2d':
        cf = ax.contourf(xvec, yvec, W_plot, num_colors, norm=norm_, cmap=cmap)
    elif projection == '3d':
        X, Y = np.meshgrid(xvec, xvec)
        # Use masked W (not W0) so masked regions become holes on the surface
        cf = ax.plot_surface(X, Y, W_plot, rstride=5, cstride=5, linewidth=0.5, norm=norm_, cmap=cmap)
    else:
        raise ValueError('Unexpected value of projection keyword argument.')

    if xvec is not yvec:
        ax.set_ylim(xvec.min(), xvec.max())

    ax.set_xlabel(r'$\rm{Re}(\alpha)$', fontsize=12)
    ax.set_ylabel(r'$\rm{Im}(\alpha)$', fontsize=12)


    # cf.set_clim(colorlim[0], colorlim[1])
    if colorbar:
        cb = _get_color_bar(fig, ax, cf, norm_)
    else:
        cb = None

    ax.set_title("Wigner function", fontsize=12)

    return fig, ax, cf, cb


def plot_plain_wigner(
    state:np.ndarray|qutip.Qobj, ax:None|Axes=None, title:str|None|Literal[False]=None, 
    with_colorbar:bool=True, with_axes:bool=True, colorlims:tuple[float, float]|Literal[True]|None=None,
    num_points:int=250, figsize=(6, 6), projection='2d', transparent_zero: float|bool = False  # new
) -> WignerOutput:
    # Inversed color-map:
    # cmap = cm.get_cmap('RdBu')
    # cmap = cmap.reversed() 
    cmap = cm.get_cmap('bwr')
    
    ## Qutip object:
    if isinstance(state, np.ndarray):
        qu_state = qutip.Qobj(state)
    elif isinstance(state, qutip.Qobj):
        qu_state = state
    else:
        raise TypeError(f"Invalid state type: {type(state)!r}")

    ## Color limits:
    if colorlims is True:
        colorlims = (-0.3, +0.3)

    ## Figure and axes:
    if ax is None:
        if projection == '2d':
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        elif projection == '3d':
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        else:
            raise ValueError('Unexpected value of projection keyword argument')

    assert isinstance(ax, plt.Axes)

    fig = ax.get_figure()

    ## plot:
    fig, ax, cf, cb = _plot_wigner(
        qu_state, fig=fig, ax=ax,
        cmap=cmap, colorbar=with_colorbar, colorlims=colorlims,
        num_points=num_points, projection=projection,
        transparent_zero=transparent_zero  # pass through
    )
    ax.grid(True)

    ## Beautify Colorbar ticks:
    if with_colorbar:
        ticks = cb.get_ticks()
        ticks = [x for i, x in enumerate(ticks) if i%2==0]  # only even numbers
        ticks = [round(x, 3) for x in ticks]  # fewer digits
        cb.set_ticks(ticks)


    # axes
    if not with_axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False
        ) #


        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelbottom=False
        ) #
        
    # Add title:
    if title is None:
        pass
    elif title is False:
        ax.set_title("")
    else:
        ax.set_title(title)

    ## Delete colors if state is an empty state:
    if np.allclose(qu_state.full(), 0.0):
        cf.remove()
        if cb is not None:
            cb.ax.clear()

    return WignerOutput(
        fig=fig, 
        ax=ax, 
        cf=cf,
        cb=cb
    )
        

def qutip_wigner_plot(q:qutip.Qobj, ax:None|Axes=None, num_points:int=250, num_colors:int=251) -> WignerOutput:
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    elif isinstance(ax, Axes):
        fig = ax.get_figure()
    assert isinstance(ax, plt.Axes)


    # Define the range for the plot
    x = np.linspace(-5, 5, num_points)
    y = np.linspace(-5, 5, num_points)
    X, Y = np.meshgrid(x, y)

    # Compute the Wigner function
    W = qutip.wigner(q, x, y)

    # Plot the Wigner function
    cf = ax.contourf(X, Y, W, num_colors, cmap='RdBu_r')
    cb = fig.colorbar(cf, ax=ax)
    ax.set_title('Wigner Function')
    ax.set_xlabel('x')
    ax.set_ylabel('p')


    return WignerOutput(
        fig=fig, 
        ax=ax, 
        cf=cf,
        cb=cb
    )



def _get_color_bar(fig:Figure, ax:Axes, cf:ContourSet, norm_=None) -> Colorbar:
    ## Check if fig already has a colorbar:
    for ax_ in fig.get_axes():
        if ax_._label == '<colorbar>':
            # Check if this colorbar is associated with this ax:
            if ax in ax_.child_axes:
                # Apply the color mapping to the existing colorbar:
                cb = fig.colorbar(cf, ax=ax, cax=ax_)
                break

    ## Create a new colorbar:   
    else:
        cb = fig.colorbar(cf, ax=ax)
        ## Give the axis of this cb and association:
        ax_ = cb.ax
        ax_.child_axes.append(ax)
    
    return cb

def _set_ndarray_within_limits(arr:np.ndarray, lims:tuple[int, int]) -> None:
    with np.nditer(arr, op_flags=['readwrite']) as it:
        for x in it:
            if x<lims[0]:
                x[...] = lims[0]
            elif x>lims[1]:
                x[...] = lims[1]



## ================== Bloch-sphere Wigner function ================== ##

        

def plot_wigner_bloch_sphere(
    rho:np.matrix, 
    num_points:int=100, 
    ax:Axes3D=None,  # type: ignore
    colorbar_ax:Axes|None=None, 
    warn_imaginary_part:bool=False, 
    title:str=None, 
    with_colorbar:bool=True,
    with_axes_arrows:bool=True,
    with_light_source:bool=True,
    alpha:float=0.2,
    view_elev:float=-40,
    view_azim:float=45,
    view_roll:float=0,
) :
    # Constants:
    radius = 1

    # Check inputs:
    if ax is None:
        fig = plt.figure(figsize=(10,8))
        ax : Axes3D = fig.add_subplot(111, projection='3d') # type: ignore
    elif isinstance(ax, Axes3D):     # type: ignore
        fig = ax.figure
    else:
        raise TypeError(f"No a supproted `ax` of type '{type(ax)}'")
        
    # Basic data:
    num_atoms = rho.shape[0] - 1
    phi = np.linspace(0, 2 * np.pi, 2 * num_points)
    theta = np.linspace(0, np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    X = np.sin(theta) * np.cos(phi) * radius
    Y = np.sin(theta) * np.sin(phi) * radius
    Z = np.cos(theta) * radius
    W = np.zeros(np.shape(phi))
    j = num_atoms / 2

    # Basic functions:
    floor = math.floor
    
    # Iterate:
    k_vals = np.linspace(0, 2*j, floor(2*j + 1))
    m_vals = np.linspace(-j, j, floor(2 * j + 1))

    for k in ProgressBar(k_vals, prefix="calculating wigner-function... ") :
        
        for q in np.linspace(-k, k, floor(2 * k + 1)):

            if q >= 0:
                Ykq = sph_harm(q, k, phi, theta)
            else:
                Ykq = (-1)**q*np.conj(sph_harm(-q, k, phi, theta))
            Gkq = 0
            
            for m1, m2 in itertools.product(m_vals, repeat=2):                
                if -m1 + m2 + q == 0:
                    tracem1m2 = rho[floor(m1 + j), floor(m2 + j)]
                    # convert from numpy float to float native float:
                    j, k, q, m1, m2 = map(float, (j, k, q, m1, m2))  
                    wig_sym = wigner_3j(j, k, j, -m1, q, m2)
                    wig_val = np.conj(complex(wig_sym))
                    Gkq = Gkq + tracem1m2 * np.sqrt(2 * k + 1) * (-1) ** (j - m1) * wig_val
                    
            W = W + Ykq * Gkq;

    if warn_imaginary_part and ( np.max(abs(np.imag(W))) > 1e-3 ):
        print('The wigner function has non negligible imaginary part ', str(np.max(abs(np.imag(W)))))
    W = np.real(W)

    ## Set colors on the spectrum red to blue:
    normalized_face_values = W / np.max(np.abs(W))
    face_colors = cm.bwr(normalized_face_values / 2 + 0.5)
    ## Adjust opacity values:
    alpha_func = lambda normalized_face_value : _face_value_function(normalized_face_value, alpha)
    for ind_, j in np.ndindex(normalized_face_values.shape):
        face_colors[ind_,j,3] = alpha_func(normalized_face_values[ind_,j])

    ## Light Source:
    if with_light_source:
        lightsource = LightSource(azdeg=view_azim, altdeg=view_elev)
    else:
        lightsource = None

    surface_plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=face_colors, lightsource=lightsource)
    if alpha < 0.6:
        surface_plot.set_edgecolors('none')  # transparent edges

    m = cm.ScalarMappable(cmap=cm.bwr)
    m.set_array(normalized_face_values)
    m.set_clim(-min(np.max(np.abs(W)),2), min(np.max(np.abs(W)),2))
    

    assert isinstance(ax, Axes)

    # Set title:
    if title is not None:
        ax.set_title(title, y=1.10)
    else:
        ax.set_title('$W(\\theta,\phi)$', fontsize=16, y=0.95)
        
    # Set "sphere orientation":
    ax.view_init(elev=view_elev, azim=view_azim, roll=view_roll)   

    # Color bar:
    if with_colorbar:
        if colorbar_ax is None:
            colorbar = plt.colorbar(mappable=m, shrink=0.5, ax=ax)
        else:
            colorbar = plt.colorbar(m, ax=ax, cax=colorbar_ax, shrink=0.5)        

    # xyz axes:
    if with_axes_arrows:
        for ind_, str_ in enumerate(['x', 'y', 'z']):
            xyz = [0, 0, 0]
            xyz[ind_] = 1.3
            quiver_inputs = [0,0,0]+xyz
            arrow = ax.quiver(
                *quiver_inputs,  length=1.2, arrow_length_ratio=0.1, zorder=100+ind_, alpha=0.5
            )
            arrow.set_capstyle
            arrow.set_linewidth(4)
            xyz[ind_] = 1.7
            ax.text(*xyz, str_, font=dict(size=18))
    

    # Turn off the axis planes
    ax.set_axis_off()

    return surface_plot


def _derive_city_colorbar_position(ax:Axes)->tuple[float,...]:

    # Pre-decided relative position
    left0, bottom0, width0, height0 = 1.10, 0.1, 0.04, 0.8
    
    # get relative place of ax within figure:
    # [[xmin, ymin], [xmax, ymax]]   
    bbox = ax.get_position()
    x0 = bbox.x0
    x1 = bbox.x1
    y0 = bbox.y0
    y1 = bbox.y1    
    w  = x1 - x0
    h  = y1 - y0

    width  = width0*w
    height = height0*h
    left   = left0*w + x0
    bottom = bottom0*h + y0

    return left, bottom, width, height


def plot_city(mat:Union[np.matrix, np.array], title:Optional[str]=None, ax:Axes3D|None=None):
    # Check input type:
    if isinstance(mat, np.matrix):
        mat = np.array(mat)
    assert len(mat.shape)==2
    assert mat.shape[0]==mat.shape[1]

    # Define common symbols:
    pi = np.pi

    n = np.size(mat)
    xpos, ypos = np.meshgrid(range(mat.shape[0]), range(mat.shape[1]))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(n)
    dx = dy = 0.8 * np.ones(n)
    Mvec = mat.flatten()
    dz = abs(Mvec).tolist()

    # make small numbers real, to avoid random colors
    idx = np.where(abs(Mvec) < 0.001)
    Mvec[idx] = abs(Mvec[idx])

    # Define colors:
    phase_min = -pi
    phase_max = pi
    norm = mpl.colors.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()
    colors = cmap(norm(np.angle(Mvec)))

    if ax is None:
        fig = plt.figure()
        ax = _axes3D(fig, azim=-35, elev=35)
    else:
        fig = ax.figure

    ## Plot:
    plot = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    if title is not None:
        ax.set_title(title, y=0.95)

    # x axis
    xtics = +0.5 + np.arange(mat.shape[0])
    ax.xaxis.set_major_locator(plt.FixedLocator(xtics))
    ax.tick_params(axis='x', labelsize=12)

    # y axis
    ytics = +0.5 + np.arange(mat.shape[1])
    ax.yaxis.set_major_locator(plt.FixedLocator(ytics))
    ax.tick_params(axis='y', labelsize=12)

    # z axis
    ax.set_zlim3d([0, 1])  # use min/max

    # Labels:
    M = mat.shape[0]
    step_size = M//5
    if step_size==0: step_size=1
    m_range = range(0, M, step_size)
    pos_range = range(0, mat.shape[0], step_size) 
    plt.sca(ax)
    plt.xticks( pos_range, [ f"{m}" for m in m_range] )
    plt.yticks( pos_range, [ f"{m}" for m in m_range] )

    ## Colorbar:
    left, bottom, width, height = _derive_city_colorbar_position(ax)    
    cax = plt.axes([left, bottom, width, height])
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    clr_bar = plt.colorbar(ax=ax, cax=cax, mappable=mappable )
    clr_bar.set_ticks([-pi, -pi/2, 0, pi/2, pi])
    clr_bar.set_ticklabels( [r"$-\pi$", r"$-\dfrac{\pi}{2}$", 0, r"$\dfrac{\pi}{2}$", r'$\pi$'] )
    
    return fig, ax, plot



# ==================================================================================== #
#|                               Helper Types                                         |#
# ==================================================================================== #
@dataclass
class ViewingAngles():
     elev : float = -40
     azim : float =  45
     roll : float =  0

@dataclass
class BlochSphereConfig():
    viewing_angles : ViewingAngles = field( default_factory=ViewingAngles )
    alpha_min : float = 1.0
    resolution : int = 100


    
class MatterStatePlot():

    _separate_colorbar_axis : ClassVar[bool] = True
    BlochSphereConfig : TypeAlias = BlochSphereConfig

    def __init__(
        self, 
        bloch_sphere_config:BlochSphereConfig=BlochSphereConfig(), 
        state:Optional[np.matrix|np.ndarray|Qobj]=None, 
        horizontal:bool=True,
        draw_now:bool=False, 
        title:str=""
    ) -> None:
        state = _convert_qobj_to_np_type(state)
        plt_objects = MatterStatePlot._init_figure(horizontal=horizontal)
        self.axis_bloch_sphere : Axes3D = plt_objects['ax_bloch_sphere']
        self.axis_bloch_sphere_colorbar : Axes = plt_objects['ax_color_bar']
        self.axis_block_city : Axes3D = plt_objects['ax_block_city']
        self.figure : Figure = plt_objects['fig']
        self.bloch_sphere_config : BlochSphereConfig = bloch_sphere_config
        self.horizontal : bool = horizontal
        if state is not None:
            self.update(state, title=title, draw_now=draw_now)
    
    def update(
        self, 
        state:np.matrix|np.ndarray, 
        title:Optional[str]=None, 
        score_str:Optional[str]=None, 
        fontsize:int=16, 
        draw_now:bool=False,        
    ) -> None:
        state = _convert_qobj_to_np_type(state)
        assertions.density_matrix(state, robust_check=False)
        self.refresh_figure()
        _, city_axes ,city_plot = plot_city(state, ax=self.axis_block_city)
        plot_wigner_bloch_sphere(
            state, ax=self.axis_bloch_sphere, 
            num_points=self.bloch_sphere_config.resolution, 
            colorbar_ax=self.axis_bloch_sphere_colorbar,
            alpha=self.bloch_sphere_config.alpha_min,
            view_azim=self.bloch_sphere_config.viewing_angles.azim, 
            view_elev=self.bloch_sphere_config.viewing_angles.elev, 
            view_roll=self.bloch_sphere_config.viewing_angles.roll, 
            title=""
        )
        city_axes.set_zorder(2)
        if title is not None:
            self.set_title(title, fontsize=fontsize)            
        if draw_now:
            matplotlib_support.draw_now()
        if score_str is not None:
            self.axis_bloch_sphere.text(x=-0.5 ,y=-0.5, z=-2, s=score_str, fontsize=12)
    
    def set_title(self, title:str, /, *, fontsize:int=16)->None:
        self.figure.suptitle(title, fontsize=fontsize)

    def close(self) -> None:
        plt.close(self.figure)
            
    def refresh_figure(self) -> dict :
        plt.figure(self.figure.number)
        plt.clf()
        plt_objects = self.__class__._init_figure(self.figure, horizontal=self.horizontal)
        self.axis_bloch_sphere : Axes3D = plt_objects['ax_bloch_sphere']
        self.axis_bloch_sphere_colorbar : Axes = plt_objects['ax_color_bar']
        self.axis_block_city : Axes3D = plt_objects['ax_block_city']
        self.figure : Figure = plt_objects['fig']
        return plt_objects
        
    @classmethod
    def _init_figure(cls, fig:Optional[Figure]=None, horizontal:bool=True) -> dict[str, Axes|Axes3D|Figure|None]:
        # Control
        separate_colorbar_axis : bool = MatterStatePlot._separate_colorbar_axis
        # fig:
        if fig is None:
            if horizontal:
                fig = plt.figure(figsize=(10,5))
            else:
                fig = plt.figure(figsize=(5 ,9))
        # Create axes:
        ax_block_city = _axes3D(fig)
        if horizontal:
            ax_bloch_sphe = fig.add_subplot(1,2,1, projection='3d')        
        else:
            ax_bloch_sphe = fig.add_subplot(2,1,1, projection='3d')        

        if separate_colorbar_axis:
            if horizontal:                # [left, bottom, width, height]
                ax_color_bar = fig.add_axes([0.00, 0.10,  0.02, 0.7])
            else:
                ax_color_bar = fig.add_axes([0.88, 0.55,  0.03, 0.4])
        else:
            ax_color_bar = None
            
        # set dims:
        if horizontal:                    # [[ xmin, ymin], [xmax, ymax]]   
            ax_bloch_sphe.set_position(Bbox([[ 0.00,-0.20], [0.55, 1.10]])) 
            ax_block_city.set_position(Bbox([[ 0.45, 0.00], [0.93, 0.90]]))  
        else:
            ax_bloch_sphe.set_position(Bbox([[-0.15, 0.40], [1.00, 1.00]])) 
            ax_block_city.set_position(Bbox([[ 0.00, 0.00], [0.84, 0.55]]))  
        # Return:
        res = dict(
            fig=fig,
            ax_bloch_sphere=ax_bloch_sphe,
            ax_color_bar=ax_color_bar ,
            ax_block_city=ax_block_city
        )
        return res



# ==================================================================================== #
#|                          Inner Helper Functions                                    |#
# ==================================================================================== #
# For function version detection:
from packaging.version import parse as parse_version

if parse_version(mpl.__version__) >= parse_version('3.4'):
    def _axes3D(fig, *args, **kwargs):
        ax = Axes3D(fig, *args, auto_add_to_figure=False, **kwargs)
        return fig.add_axes(ax)
else:
    def _axes3D(*args, **kwargs):
        return Axes3D(*args, **kwargs)


def _sigmoid(x, center_x:float, steepness:float):
    exponent = -steepness*((x-center_x))
    return 1/( 1 + np.exp(exponent) )

def _face_value_function( value:float, lowest_alpha:float) -> float:
    x = abs(value)
    y = max(_sigmoid(x, 0.1, 10), 0)
    return (1-lowest_alpha)*y + lowest_alpha









## ================== MAIN ================== ##


def main():
    q = qutip.Qobj( np.array([[1, 0], [0, 1]]) )
    qutip_wigner_plot(q)
    matplotlib_support.draw_now()
    print("Done.")


if __name__ == "__main__":
    main()