from tvb.simulator.models.base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class MontbrioT(ModelNumbaDfun):
        
    I = NArray(
        label=":math:`I`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""???"""
    )    
        
    Delta = NArray(
        label=":math:`Delta`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Vertical shift of the configurable nullcline."""
    )    
        
    alpha = NArray(
        label=":math:`alpha`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc=""":math:`\alpha` ratio of effect between long-range and local connectivity."""
    )    
        
    s = NArray(
        label=":math:`s`",
        default=numpy.array([0.0]),
        domain=Range(lo=-15.0, hi=15.0, step=0.01),
        doc="""QIF membrane reversal potential."""
    )    
        
    k = NArray(
        label=":math:`k`",
        default=numpy.array([0.0]),
        domain=Range(lo=-15.0, hi=15.0, step=0.01),
        doc="""Switch for the terms specific to Coombes model."""
    )    
        
    J = NArray(
        label=":math:`J`",
        default=numpy.array([15.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the slow variable to the firing rate variable."""
    )    
        
    eta = NArray(
        label=":math:`eta`",
        default=numpy.array([-5.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the firing rate variable to itself"""
    )    
        
    Gamma = NArray(
        label=":math:`Gamma`",
        default=numpy.array([0.0]),
        domain=Range(lo=0., hi=10.0, step=0.1),
        doc="""Derived from eterogeneous currents and synaptic weights (see Montbrio p.12)."""
    )    
        
    gamma = NArray(
        label=":math:`gamma`",
        default=numpy.array([666]),
        domain=Range(lo=-2.0, hi=2.0, step=0.1),
        doc="""Constant parameter to reproduce FHN dynamics where excitatory input currents are negative. It scales both I and the long range coupling term."""
    )    
        
    ccr_mha = NArray(
        label=":math:`ccr_mha`",
        default=numpy.array([[10.          9.85294118  9.70588235  9.55882353  9.41176471  9.26470588
  9.11764706  8.97058824  8.82352941  8.67647059  8.52941176  8.38235294
  8.23529412  8.08823529  7.94117647  7.79411765  7.64705882  7.5
  7.35294118  7.20588235  7.05882353  6.91176471  6.76470588  6.61764706
  6.47058824  6.32352941  6.17647059  6.02941176  5.88235294  5.73529412
  5.58823529  5.44117647  5.29411765  5.14705882  5.          4.85294118
  4.70588235  4.55882353  4.41176471  4.26470588  4.11764706  3.97058824
  3.82352941  3.67647059  3.52941176  3.38235294  3.23529412  3.08823529
  2.94117647  2.79411765  2.64705882  2.5         2.35294118  2.20588235
  2.05882353  1.91176471  1.76470588  1.61764706  1.47058824  1.32352941
  1.17647059  1.02941176  0.88235294  0.73529412  0.58823529  0.44117647
  0.29411765  0.14705882]]),
        domain=Range(lo=1.0, hi=2.0, step=3.0),
        doc="""cell count/region."""
    )    
        
    scr_mha = NArray(
        label=":math:`scr_mha`",
        default=numpy.array([[1.         0.98529412 0.97058824 0.95588235 0.94117647 0.92647059
 0.91176471 0.89705882 0.88235294 0.86764706 0.85294118 0.83823529
 0.82352941 0.80882353 0.79411765 0.77941176 0.76470588 0.75
 0.73529412 0.72058824 0.70588235 0.69117647 0.67647059 0.66176471
 0.64705882 0.63235294 0.61764706 0.60294118 0.58823529 0.57352941
 0.55882353 0.54411765 0.52941176 0.51470588 0.5        0.48529412
 0.47058824 0.45588235 0.44117647 0.42647059 0.41176471 0.39705882
 0.38235294 0.36764706 0.35294118 0.33823529 0.32352941 0.30882353
 0.29411765 0.27941176 0.26470588 0.25       0.23529412 0.22058824
 0.20588235 0.19117647 0.17647059 0.16176471 0.14705882 0.13235294
 0.11764706 0.10294118 0.08823529 0.07352941 0.05882353 0.04411765
 0.02941176 0.01470588]]),
        domain=Range(lo=2.0, hi=3.0, step=4.0),
        doc="""cell count/region."""
    )    
        
    fbr_mha = NArray(
        label=":math:`fbr_mha`",
        default=numpy.array([[100.          98.52941176  97.05882353  95.58823529  94.11764706
  92.64705882  91.17647059  89.70588235  88.23529412  86.76470588
  85.29411765  83.82352941  82.35294118  80.88235294  79.41176471
  77.94117647  76.47058824  75.          73.52941176  72.05882353
  70.58823529  69.11764706  67.64705882  66.17647059  64.70588235
  63.23529412  61.76470588  60.29411765  58.82352941  57.35294118
  55.88235294  54.41176471  52.94117647  51.47058824  50.
  48.52941176  47.05882353  45.58823529  44.11764706  42.64705882
  41.17647059  39.70588235  38.23529412  36.76470588  35.29411765
  33.82352941  32.35294118  30.88235294  29.41176471  27.94117647
  26.47058824  25.          23.52941176  22.05882353  20.58823529
  19.11764706  17.64705882  16.17647059  14.70588235  13.23529412
  11.76470588  10.29411765   8.82352941   7.35294118   5.88235294
   4.41176471   2.94117647   1.47058824]]),
        domain=Range(lo=3.0, hi=4.0, step=5.0),
        doc="""cell count/region."""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0., 2.0]), 
				 "V": numpy.array([-2.0, 1.5])},
        doc="""state variables"""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"r": numpy.array([0.0, inf])},
    )
    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('r', 'V', ),
        default=('r', 'V', ),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator."
    )

    state_variables = ['r', 'V']

    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_MontbrioT(vw_, c_, self.I, self.Delta, self.alpha, self.s, self.k, self.J, self.eta, self.Gamma, self.gamma, self.ccr_mha, self.scr_mha, self.fbr_mha, local_coupling)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])], '(n),(m)' + ',()'*13 + '->(n)', nopython=True)
def _numba_dfun_MontbrioT(vw, coupling, I, Delta, alpha, s, k, J, eta, Gamma, gamma, ccr_mha, scr_mha, fbr_mha, local_coupling, dx):
    "Gufunc for MontbrioT model equations."

    r = vw[0]
    V = vw[1]

    Coupling_global = alpha * coupling[0]
    Coupling_local = (1-alpha) * local_coupling * r
    Coupling_Term = Coupling_global + Coupling_local

    dx[0] = Delta / pi + 2 * V * r - k * r**2 + Gamma * r / pi
    dx[1] = V**2 - pi**2 * r**2 + eta + (k * s + J) * r - k * V * r + gamma * I + Coupling_Term
            