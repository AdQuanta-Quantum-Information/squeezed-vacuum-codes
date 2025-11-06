from src.quantum.states._base import _ChainedOperator, MeasurementTensorProduct, IdentityOperator, Operator, _TensorProductSum
from src.quantum.states.bosonic.squeezed import SqueezedCoherentState, SqueezingOperator
from src.quantum.states.bosonic.coherent import Displacement
from src.quantum.states.bosonic.qutip_light import LightState
from src.utils import strings, tuples, lists
from src.quantum.states._base import Operator, StatesTensorProduct, IdentityOperator, StatesSum
from src.quantum.states.qudit import QuditState, QuditSum, QuditBasis 
from src.quantum.states.qudit import QuditX, QuditY, QuditH, QuditPhase

from typing import Self, cast, Tuple, Type, TypeAlias, Generic, TypeVar

_LightRepresentation = TypeVar("_LightRepresentation", SqueezedCoherentState, LightState)
_QutritAndLightTypeTuple : TypeAlias = Tuple[QuditState, _LightRepresentation]



class _BaseQutritAndLightState(StatesTensorProduct[_QutritAndLightTypeTuple], Generic[_LightRepresentation]):
    """A state that is a tensor product of a qutrit and a light state."""

    __sum_class : Type[StatesSum] = "QutritAndLightSuperposition"   

    qutrit_basis : QuditBasis = QuditBasis(num_digits=3, _possible_values={-1, 0, 1})

    @classmethod
    def cast_from_tensor_product(cls, tensor_product:StatesTensorProduct[_QutritAndLightTypeTuple]) -> Self:
        if isinstance(tensor_product, _TensorProductSum):
            states = []
            for state_ in tensor_product:
                casted_state = cls.cast_from_tensor_product(state_)
                states.append(casted_state)
            return lists.sum(states)

        elif isinstance(tensor_product, StatesTensorProduct):
            # Get and check parts:
            assert len(tensor_product) == 2, f"Expected a tensor product with 2 components, but got {tensor_product}."
            qutrit = tensor_product[0]
            light = tensor_product[1]
            weight = tensor_product.weight
            assert isinstance(qutrit, QuditState)
            assert isinstance(light, SqueezedCoherentState)
            # Create the state:
            return cls((qutrit, light))*weight
        
        else:
            raise TypeError(f"Unexpected type for tensor_product: {tensor_product!r}.")

    @classmethod
    def qutrit_measurement(cls) -> MeasurementTensorProduct[_QutritAndLightTypeTuple, Tuple[_LightRepresentation]]:
        M = cls.qutrit_basis.measurement()
        I = IdentityOperator[_LightRepresentation]()
        return M & I

    @classmethod
    def zero_state(cls) -> Self:
        cls.qutrit_basis(value=0)
        qutrit = QuditState(value=0, num_digits=3, _possible_values={-1, 0, 1})
        light = SqueezedCoherentState(displacement=0.0, squeezing=0.0)
        return cls((qutrit, light))

    @property
    def qutrit(self) -> QuditState:
        return cast(QuditState, self.states[0])
    
    @property
    def light(self) -> SqueezedCoherentState:
        return cast(SqueezedCoherentState, self.states[1])

    def validate(self) -> None:        
        # The super validations for tensor product states:
        super().validate()
        ## Specific validation of the QutritAndLightState:
        
        if len(self) != 2:
            raise ValueError(f"Expected a state with 2 components, but got {self}.")
        qutrit = self.qutrit
        light  = self.light
        assert self[0] is qutrit
        assert self[1] is light
        if not isinstance(qutrit, QuditState):
            raise ValueError(f"Expected 1st state to be a qutrit state of type QuditState, but got {qutrit}.")
        if qutrit.num_digits != 3:
            raise ValueError(f"Expected 1st state to be a qutrit state with 3 digits, but got {qutrit}.")
        if not isinstance(light, SqueezedCoherentState):
            raise ValueError(f"Expected 2nd state to be a light state of type SqueezedCoherentState, but got {light}.")


    @classmethod
    def qutrit_span(cls) -> set[QuditState]:
        return {cls.qutrit_basis(v) for v in cls.qutrit_basis._possible_values}

    def __repr__(self, _include_address:bool=False) -> str:
        s = "Qutrit&Light: "
        s += self.__str__()
        if _include_address:
            s += f" at ({id(self)})"
        return s


class QutritAndLightSuperposition(StatesSum[_BaseQutritAndLightState]):
    __state_type = _BaseQutritAndLightState
    
    def _class_name_str(self) -> str:
        return f"Qutrit&Light"  


class QutritAndSqueezedCoherentLight(_BaseQutritAndLightState[SqueezedCoherentState]): ...


class QutritAndLightState(_BaseQutritAndLightState[LightState]): ...




class _ControlledOperationOnLight(Operator[_BaseQutritAndLightState[_LightRepresentation]]):
    """An operator that applies a squeezing operation to a light state, in accordance with the state of a qutrit."""
    __slots__ = ("strength")

    _operator : Type[Operator[_LightRepresentation]]


    def __init__(self, strength:complex) -> None:
        self.strength : complex = strength

    def _apply_on_ket_no_weight(self, state:_BaseQutritAndLightState|StatesTensorProduct[_QutritAndLightTypeTuple]) -> _BaseQutritAndLightState:
        ## Check input:
        if isinstance(state, _BaseQutritAndLightState):
            state.validate()
        elif isinstance(state, StatesTensorProduct):
            state = _BaseQutritAndLightState.cast_from_tensor_product(state)
        else:
            raise TypeError(f"Unexpected type for state: {state!r}.")        
        
        ## unpack data:
        qutrit = state.qutrit
        light = state.light
        q = qutrit.value

        ## if q is 0, then the state is unchanged:
        if q == 0:
            return state
        
        ## Get the squeezing operator according to the qutrit value:
        elif q == -1:
            light_operator = self._operator(-self.strength)
        elif q == +1:
            light_operator = self._operator(+self.strength)
        else:
            raise ValueError(f"Unexpected qutrit value: {q!r}.")

        ## Apply the squeezing operation:
        squeezed_light = light_operator @ light
        new_state = _BaseQutritAndLightState[_LightRepresentation]((qutrit, squeezed_light))
        return new_state
    

    def __matmul__(self, other: _BaseQutritAndLightState[_LightRepresentation]|StatesTensorProduct[_QutritAndLightTypeTuple]) -> _BaseQutritAndLightState[_LightRepresentation]:  # called in the statement `self @ other`
        if isinstance(other, StatesTensorProduct):
            other = _BaseQutritAndLightState.cast_from_tensor_product(other)
        return super().__matmul__(other)
    
    def __repr__(self) -> str:
        gamma_str = strings.simplified_complex_str(self.strength)
        return f"CS({gamma_str})"



class ControlledDisplacement(_ControlledOperationOnLight):
    _operator = Displacement

class ControlledSqueezing(_ControlledOperationOnLight):
    _operator = SqueezingOperator