import theano
import theano.tensor as T
import numpy as np

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty, Property
from .base import MinibatchGradientDescent
from hessian_utils import find_hessian_and_gradient
from neupy.algorithms.utils import (parameters2vector, count_parameters,
                                    iter_parameters, setup_parameter_updates)

__all__ = ('Momentum',)


class Momentum(MinibatchGradientDescent):
    """
    Momentum algorithm for :network:`GradientDescent` optimization.

    Parameters
    ----------
    momentum : float
        Control previous gradient ratio. Defaults to ``0.9``.
    nesterov : bool
        Instead of classic momentum computes Nesterov momentum.
        Defaults to ``False``.
    {MinibatchGradientDescent.Parameters}

    Attributes
    ----------
    {MinibatchGradientDescent.Attributes}

    Methods
    -------
    {MinibatchGradientDescent.Methods}

    Examples
    --------
    Simple example

    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> mnet = algorithms.Momentum(
    ...     (2, 3, 1),
    ...     verbose=False
    ... )
    >>> mnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    momentum = ProperFractionProperty(default=0.9)
    nesterov = Property(default=False, expected_type=bool)

    def init_layers(self):
        super(Momentum, self).init_layers()
        for layer in self.layers:
            for parameter in layer.parameters:
                parameter_shape = T.shape(parameter).eval()
                parameter.prev_param_delta = theano.shared(
                    name="prev_param_delta_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )


    def init_param_updates(self, layer, parameter):
        n_parameters = count_parameters(self)
        self.variables.hessian = theano.shared(
            value=asfloat(np.zeros((n_parameters, n_parameters))),
            name='hessian_inverse')
        step = self.variables.step
        gradient = T.grad(self.variables.error_func, wrt=parameter)

        prev_param_delta = parameter.prev_param_delta
        parameter_delta = self.momentum * prev_param_delta - step * gradient

        if self.nesterov:
            parameter_delta = self.momentum * parameter_delta - step * gradient
# modified for hessian

        n_parameters = count_parameters(self)
        parameters = list(iter_parameters(self))
        param_vector = parameters2vector(self)
#        penalty_const = asfloat(self.penalty_const)

        hessian_matrix, full_gradient = find_hessian_and_gradient(
            self.variables.error_func, parameters
        )
#        hessian_inverse = T.nlinalg.matrix_inverse(
#            hessian_matrix + 0.02 * T.eye(n_parameters)
#        )

# Modified fo hessian

        return [
            (parameter, parameter + parameter_delta),
            (prev_param_delta, parameter_delta),(self.variables.hessian, hessian_matrix)
        ]
