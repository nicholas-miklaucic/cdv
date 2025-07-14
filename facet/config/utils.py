from typing import Any, Callable, Optional

import jax.numpy as jnp
import pyrallis
from flax import linen as nn
from pyrallis.fields import field

from facet import layers
from facet.config.common import dataclass
from facet.layers import Identity, LazyInMLP


@dataclass
class Layer:
    """Serializable layer representation. Works for any named layer in layers.py or flax.nn."""

    # The name of the layer.
    name: str

    def build(self) -> Callable:
        """Makes a new layer with the given values, or returns the function if it's a function."""
        if self.name == 'Identity':
            return Identity()

        for module in (nn, layers, jnp):
            if hasattr(module, self.name):
                layer = getattr(module, self.name)
                if isinstance(layer, nn.Module):
                    return getattr(module, self.name)()
                else:
                    # something like relu
                    return layer

        msg = f'Could not find {self.name} in flax.linen or cdv.layers'
        raise ValueError(msg)


@dataclass
class MLPConfig:
    """Settings for MLP configuration."""

    # Inner dimensions for the MLP.
    inner_dims: list[int] = field(default_factory=list)

    # Inner activation.
    activation: str = 'swish'

    # Final activation.
    final_activation: str = 'Identity'

    # Output dimension. None means the same size as the input.
    out_dim: Optional[int] = None

    # Dropout.
    dropout: float = 0.0

    # Whether to add residual.
    residual: bool = False

    # Number of heads, for mixer.
    num_heads: int = 1

    # Whether to use a bias. Also applies to LayerNorm.
    use_bias: bool = False

    # normalization: 'layer', 'weight', 'dytanh', 'none'
    normalization: str = 'layer'

    def build(self) -> LazyInMLP:
        """Builds the head from the config."""
        return LazyInMLP(
            inner_dims=self.inner_dims,
            out_dim=self.out_dim,
            inner_act=Layer(name=self.activation).build(),
            final_act=Layer(name=self.final_activation).build(),
            dropout_rate=self.dropout,
            residual=self.residual,
            use_bias=self.use_bias,
            normalization=self.normalization,
        )


class Constant:
    """
    A value that can only hold a single literal. Used as a tag for union types so Pyrallis can
    decode them properly and parsing configs never has any strange surprises.
    """

    value: Any = None

    @classmethod
    def _decode(cls, x):
        if x == cls.value:
            return x
        else:
            raise ValueError(f'{x} != {cls.value}')

    def _encode(self):
        return self.value


pyrallis.decode.register(Constant, lambda t, x: t._decode(x), include_subclasses=True)
pyrallis.encode.register(Constant, Constant._encode)


def Const(val) -> type[Constant]:
    class ConstantImpl(Constant):
        value = val

    return ConstantImpl
