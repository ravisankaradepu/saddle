import theano
import theano.typed_list
import theano.tensor as T
def find_hessian_and_gradient(error_function, parameters):
    """
    Find Hessian and gradient for the Neural Network cost function.

    Parameters
    ----------
    function : Theano function
    parameters : list
        List of all Neural Network parameters.

    Returns
    -------
    Theano function
    """
    n_parameters = T.sum([parameter.size for parameter in parameters])
    gradients = T.grad(error_function, wrt=parameters)
    full_gradient = T.concatenate([grad.flatten() for grad in gradients])

    def find_hessian(i, full_gradient, *parameters):
        second_derivatives = []
        g = full_gradient[i]
        for parameter in parameters:
            second_derivative = T.grad(g, wrt=parameter)
            second_derivatives.append(second_derivative.flatten())

        return T.concatenate(second_derivatives)

    hessian, _ = theano.scan(
        find_hessian,
        sequences=T.arange(n_parameters),
        non_sequences=[full_gradient] + parameters,
    )
    hessian_matrix = hessian.reshape((n_parameters, n_parameters))
    return hessian_matrix, full_gradient
