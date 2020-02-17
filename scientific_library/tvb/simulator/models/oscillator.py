from .base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class Generic2dOscillator(ModelNumbaDfun):

        
    tau = NArray(
        label=":math:`tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=5.0, step=0.01),
        doc="""A time-scale hierarchy can be introduced for the state variables :math:`V` and :math:`W`. Default parameter is 1, which means no time-scale hierarchy."""
    )    
        
    I = NArray(
        label=":math:`I`",
        default=numpy.array([0.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Baseline shift of the cubic nullcline"""
    )    
        
    a = NArray(
        label=":math:`a`",
        default=numpy.array([-2.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Vertical shift of the configurable nullcline"""
    )    
        
    b = NArray(
        label=":math:`b`",
        default=numpy.array([-10.0]),
        domain=Range(lo=-20.0, hi=15.0, step=0.01),
        doc="""Linear slope of the configurable nullcline"""
    )    
        
    c = NArray(
        label=":math:`c`",
        default=numpy.array([0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""Parabolic term of the configurable nullcline"""
    )    
        
    d = NArray(
        label=":math:`d`",
        default=numpy.array([0.02]),
        domain=Range(lo=0.0001, hi=1.0, step=0.0001),
        doc="""Temporal scale factor. Warning: do not use it unless you know what you are doing and know about time tides."""
    )    
        
    e = NArray(
        label=":math:`e`",
        default=numpy.array([3.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the quadratic term of the cubic nullcline."""
    )    
        
    f = NArray(
        label=":math:`f`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the cubic term of the cubic nullcline."""
    )    
        
    g = NArray(
        label=":math:`g`",
        default=numpy.array([0.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.5),
        doc="""Coefficient of the linear term of the cubic nullcline."""
    )    
        
    alpha = NArray(
        label=":math:`alpha`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the slow variable to the fast variable."""
    )    
        
    beta = NArray(
        label=":math:`beta`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the slow variable to itself"""
    )    
        
    gamma = NArray(
        label=":math:`gamma`",
        default=numpy.array([1.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.1),
        doc="""Constant parameter to reproduce FHN dynamics where excitatory input currents are negative. It scales both I and the long range coupling term.."""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"V": numpy.array([-2.0, 4.0]), 
				 "W": numpy.array([-6.0, 6.0])},
        doc="""state variables"""
        )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("'V', 'W', 'V + W', 'V - W'"),
        default=("V", ),
        doc="The quantities of interest for monitoring for the generic 2D oscillator."
    )

    state_variables = ['V', 'W']

    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0, ev=numexpr.evaluate):

        V = state_variables[0, :]
        lc_0 = local_coupling * V
        W = state_variables[1, :]

        #[State_variables, nodes]
        c_0 = coupling[0, :]

        tau = self.tau
        I = self.I
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        e = self.e
        f = self.f
        g = self.g
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        derivative = numpy.empty_like(state_variables)


        ev('d * tau * (alpha * W - f * V**3 + e * V**2 + g * V + gamma * I + gamma * c_0 + lc_0)', out=derivative[0])
        ev('d * (a + b * V + c * V**2 - beta * W) / tau', out=derivative[1])

        return derivative

    def dfun(self, vw, c, local_coupling=0.0):
        lc_0 = local_coupling * vw[0, :, 0]
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_Generic2dOscillator(vw_, c_, self.tau, self.I, self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.alpha, self.beta, self.gamma, lc_0)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:],) * 16], '(n),(m)' + ',()'*13 + '->(n)', nopython=True)
def _numba_dfun_Generic2dOscillator(vw, c_0, tau, I, a, b, c, d, e, f, g, alpha, beta, gamma, lc_0, dx):
    "Gufunc for Generic2dOscillator model equations."

    V = vw[0]
    W = vw[1]

    tau = tau[0]
    I = I[0]
    a = a[0]
    b = b[0]
    c = c[0]
    d = d[0]
    e = e[0]
    f = f[0]
    g = g[0]
    alpha = alpha[0]
    beta = beta[0]
    gamma = gamma[0]
    c_0 = c_0[0]
    lc_0 = lc_0[0]


    dx[0] = d * tau * (alpha * W - f * V**3 + e * V**2 + g * V + gamma * I + gamma * c_0 + lc_0)
    dx[1] = d * (a + b * V + c * V**2 - beta * W) / tau
            