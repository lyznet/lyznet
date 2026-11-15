"""Small neural Lyapunov candidates used by learning and verification."""

from copy import deepcopy
from hashlib import sha256
from numbers import Integral

import numpy as np
import torch
from torch import nn


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("{} must be an integer.".format(name))
    if value < 1:
        raise ValueError("{} must be positive.".format(name))
    return int(value)


def _numpy_points(points, dimension):
    values = np.asarray(points)
    if values.ndim == 0 or values.shape[-1] != dimension:
        raise ValueError(
            "points must have final dimension {}.".format(dimension)
        )
    if np.iscomplexobj(values):
        raise ValueError("points must contain real values.")
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("points must contain finite values.")
    return values


def _dtype_and_device(model):
    parameter = next(model.parameters())
    return parameter.dtype, parameter.device


def _fingerprint(module, description):
    digest = sha256(repr(description).encode("utf-8"))
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class TanhNetwork(nn.Module):
    """A scalar tanh MLP with an explicit input gradient."""

    def __init__(
        self,
        state_dimension,
        hidden_layers=2,
        width=20,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.state_dimension = _positive_integer(
            state_dimension, "state_dimension"
        )
        self.hidden_layers = _positive_integer(
            hidden_layers, "hidden_layers"
        )
        self.width = _positive_integer(width, "width")
        dimensions = (
            (self.state_dimension, self.width),
            *((self.width, self.width),) * (self.hidden_layers - 1),
        )
        self.layers = nn.ModuleList(
            nn.Linear(
                input_width,
                output_width,
                device=device,
                dtype=dtype,
            )
            for input_width, output_width in dimensions
        )
        self.output_layer = nn.Linear(
            self.width, 1, device=device, dtype=dtype
        )

    def forward(self, states):
        hidden = states
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return self.output_layer(hidden)

    def value_and_gradient(self, states):
        """Return the value and row gradient using the chain rule."""

        hidden = states
        activations = []
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
            activations.append(hidden)
        values = self.output_layer(hidden)
        gradient = self.output_layer.weight.expand(states.shape[0], -1)
        for layer, activation in zip(
            reversed(self.layers), reversed(activations)
        ):
            gradient = (gradient * (1 - activation.square())) @ layer.weight
        return values, gradient


class CenteredNeuralLyapunov(nn.Module):
    r"""The frozen correction

    ``W(x)=N(x)-N(x*)-grad(N)(x*)^T(x-x*)``.
    """

    architecture = "plain"

    def __init__(self, model, centre):
        super().__init__()
        if type(model) is not TanhNetwork:
            raise TypeError("model must be exactly a TanhNetwork.")
        self.state_dimension = model.state_dimension
        model = deepcopy(model).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        dtype, device = _dtype_and_device(model)
        centre = torch.as_tensor(centre, dtype=dtype, device=device)
        if centre.shape != (self.state_dimension,):
            raise ValueError("centre has the wrong dimension.")
        with torch.no_grad():
            value, gradient = model.value_and_gradient(centre.reshape(1, -1))
        self.model = model
        self.register_buffer("centre", centre.clone())
        self.register_buffer("centre_value", value.reshape(()).clone())
        self.register_buffer("centre_gradient", gradient.reshape(-1).clone())

    def forward(self, states):
        displacement = states - self.centre
        affine = torch.sum(
            displacement * self.centre_gradient, dim=1, keepdim=True
        )
        return self.model(states) - self.centre_value - affine

    def value_and_gradient(self, states):
        values, gradients = self.model.value_and_gradient(states)
        displacement = states - self.centre
        affine = torch.sum(
            displacement * self.centre_gradient, dim=1, keepdim=True
        )
        return (
            values - self.centre_value - affine,
            gradients - self.centre_gradient,
        )

    def value(self, points):
        values = _numpy_points(points, self.state_dimension)
        dtype, device = _dtype_and_device(self.model)
        inputs = torch.as_tensor(
            values.reshape(-1, self.state_dimension),
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            output = self(inputs).reshape(values.shape[:-1])
        return output.cpu().numpy()

    def proof_fingerprint(self):
        return _fingerprint(
            self,
            (type(self).__qualname__, self.state_dimension, self.architecture),
        )


class QuadraticAnchoredNeuralLyapunov(nn.Module):
    r"""A positive neural modulation of a quadratic Lyapunov function.

    For ``y=x-x*`` and ``G=N(x)-N(x*)``, the certificate is

    ``H=scale*y.T P y*exp(G)``, ``W=tanh(H)``.
    """

    architecture = "anchored"

    def __init__(self, model, centre, matrix, scale=0.25):
        super().__init__()
        if type(model) is not TanhNetwork:
            raise TypeError("model must be exactly a TanhNetwork.")
        self.state_dimension = model.state_dimension
        centre = np.array(centre, dtype=float, copy=True)
        matrix = np.array(matrix, dtype=float, copy=True)
        if centre.shape != (self.state_dimension,):
            raise ValueError("centre has the wrong dimension.")
        if matrix.shape != (self.state_dimension, self.state_dimension):
            raise ValueError("matrix has the wrong shape.")
        if not np.array_equal(matrix, matrix.T):
            raise ValueError("matrix must be exactly symmetric.")
        if np.min(np.linalg.eigvalsh(matrix)) <= 0:
            raise ValueError("matrix must be positive definite.")
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be positive and finite.")

        model = deepcopy(model).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        dtype, device = _dtype_and_device(model)
        centre_tensor = torch.as_tensor(centre, dtype=dtype, device=device)
        matrix_tensor = torch.as_tensor(matrix, dtype=dtype, device=device)
        with torch.no_grad():
            centre_output = model(centre_tensor.reshape(1, -1))
        self.model = model
        self.register_buffer("centre", centre_tensor.clone())
        self.register_buffer("matrix", matrix_tensor.clone())
        self.register_buffer(
            "scale", torch.as_tensor(scale, dtype=dtype, device=device)
        )
        self.register_buffer(
            "centre_output", centre_output.reshape(()).clone()
        )

    def latent_and_gradient(self, states):
        neural, neural_gradient = self.model.value_and_gradient(states)
        displacement = states - self.centre
        quadratic = torch.sum(
            (displacement @ self.matrix) * displacement,
            dim=1,
            keepdim=True,
        )
        quadratic_gradient = 2 * displacement @ self.matrix
        exponential = torch.exp(neural - self.centre_output)
        latent = self.scale * quadratic * exponential
        latent_gradient = self.scale * exponential * (
            quadratic_gradient + quadratic * neural_gradient
        )
        return latent, latent_gradient

    def forward(self, states):
        latent, _ = self.latent_and_gradient(states)
        return torch.tanh(latent)

    def value_and_gradient(self, states):
        latent, latent_gradient = self.latent_and_gradient(states)
        values = torch.tanh(latent)
        return values, (1 - values.square()) * latent_gradient

    def value(self, points):
        values = _numpy_points(points, self.state_dimension)
        dtype, device = _dtype_and_device(self.model)
        inputs = torch.as_tensor(
            values.reshape(-1, self.state_dimension),
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            output = self(inputs).reshape(values.shape[:-1])
        return output.cpu().numpy()

    def proof_fingerprint(self):
        return _fingerprint(
            self,
            (type(self).__qualname__, self.state_dimension, self.architecture),
        )


class HomogeneousNeuralLyapunov(nn.Module):
    r"""The positive homogeneous function

    ``V(x)=||x||^p exp(N(x/||x||))``, with ``V(0)=0``.

    The neural network represents only the angular log shape. Positivity and
    degree-``p`` homogeneity therefore hold by construction away from the
    origin; formal verification must still establish strict decrease.
    """

    architecture = "homogeneous"

    def __init__(self, model, degree=1):
        super().__init__()
        if type(model) is not TanhNetwork:
            raise TypeError("model must be exactly a TanhNetwork.")
        self.degree = _positive_integer(degree, "degree")
        self.state_dimension = model.state_dimension
        model = deepcopy(model).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    def value_and_gradient(self, states):
        norm = torch.linalg.vector_norm(states, dim=1, keepdim=True)
        safe_norm = norm.clamp_min(torch.finfo(states.dtype).tiny)
        directions = states / safe_norm
        log_shape, angular_gradient = self.model.value_and_gradient(
            directions
        )
        values = norm.pow(self.degree) * torch.exp(log_shape)
        radial = self.degree * directions
        tangent = angular_gradient - torch.sum(
            angular_gradient * directions, dim=1, keepdim=True
        ) * directions
        gradients = values / safe_norm * (radial + tangent)
        gradients = torch.where(norm > 0, gradients, torch.zeros_like(gradients))
        return values, gradients

    def forward(self, states):
        values, _ = self.value_and_gradient(states)
        return values

    def value(self, points):
        values = _numpy_points(points, self.state_dimension)
        dtype, device = _dtype_and_device(self.model)
        inputs = torch.as_tensor(
            values.reshape(-1, self.state_dimension),
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            output = self(inputs).reshape(values.shape[:-1])
        return output.cpu().numpy()

    def proof_fingerprint(self):
        return _fingerprint(
            self,
            (
                type(self).__qualname__,
                self.state_dimension,
                self.architecture,
                self.degree,
            ),
        )


__all__ = [
    "CenteredNeuralLyapunov",
    "HomogeneousNeuralLyapunov",
    "QuadraticAnchoredNeuralLyapunov",
    "TanhNetwork",
]
