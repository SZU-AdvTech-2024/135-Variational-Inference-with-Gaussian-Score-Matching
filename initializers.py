## Example for fitting a target Multivariate Gaussian distribution with GSM and ADVI
## Variational distribution is initialized with LBFGS fit for the mean and covariance
## The progress is monitored with a Monitor class. 

## Uncomment the following lines if you run into memory issues with JAX
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

import jax.numpy as jnp
from jax import jit, grad, random
import numpyro.distributions as dist
import optax

# enable 16 bit precision for jax required for lbfgs initializer
from jax import config

config.update("jax_enable_x64", True)

from gsmvi.gsm import GSM
from gsmvi.advi import ADVI
from gsmvi.initializers import lbfgs_init
from gsmvi.monitors import KLMonitor
import gsmvi.bam as bam2


#####
def setup_model(D=10):
    # setup a Gaussian target distribution
    mean = np.random.random(D)

    L = np.random.normal(size=D ** 2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D) * 1e-3
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    return model


#
def gsm_fit(D, lp, lp_g, mean_init, cov_init, lbfgs_res, niter0, batch_size):
    print("Now fit with GSM")
    niter = niter0
    # batch_size = 2
    key = random.PRNGKey(99)
    monitor = KLMonitor(batch_size_kl=32, checkpoint=10,
                        offset_evals=lbfgs_res.nfev)  # note the offset number of evals

    gsm = GSM(D=D, lp=lp, lp_g=lp_g)
    mean_fit, cov_fit = gsm.fit(key, mean=mean_init, cov=cov_init, niter=niter, batch_size=batch_size, monitor=monitor)
    return mean_fit, cov_fit, monitor


#
def advi_fit(D, lp, lp_g, mean_init, cov_init, lbfgs_res, niter0, batch_size):
    print("\nNow fit with ADVI")
    niter = niter0
    lr = 1e-3
    # batch_size = 2
    key = random.PRNGKey(99)
    monitor = KLMonitor(batch_size_kl=32, checkpoint=10,
                        offset_evals=lbfgs_res.nfev)  # note the offset number of evals

    opt = optax.adam(learning_rate=lr)
    advi = ADVI(D=D, lp=lp)
    mean_fit, cov_fit, losses = advi.fit(key, mean=mean_init, cov=cov_init, opt=opt, batch_size=batch_size, niter=niter,
                                         monitor=monitor)
    return mean_fit, cov_fit, monitor


#
def bam_fit2(D, lp, lp_g, mean_init, cov_init, lbfgs_res, niter0, batch_size0=2):
    print("\nNow fit with BaM old")
    niter = niter0
    batch_size = batch_size0
    key = random.PRNGKey(99)
    monitor = KLMonitor(batch_size_kl=32, checkpoint=10,
                        offset_evals=lbfgs_res.nfev)  # note the offset number of evals

    regularizer = bam2.Regularizers()
    # func = lambda i: niter / (1 + i)
    # regf = regularizer.custom(func)
    # regf = regularizer.constant(niter)
    regf = regularizer.linear(niter)

    use_lowrank = False
    # use_lowrank = True

    bam = bam2.BaM(D=D, lp=lp, lp_g=lp_g, use_lowrank=use_lowrank, jit_compile=True)
    mean_fit, cov_fit = bam.fit(key, regf=regf, mean=mean_init, cov=cov_init, batch_size=batch_size, niter=niter,
                                monitor=monitor)
    return mean_fit, cov_fit, monitor


if __name__ == "__main__":
    ###
    # setup a toy Gaussia model and extracet score needed for GSM
    D = 16
    niter = 5000
    batch_size = 8
    batch_size2 = 16
    model = setup_model(D=D)

    mean, cov, var = model.loc, model.covariance_matrix, model.variance
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))

    ###
    print("Initialize with LBFGS")
    mean_init = np.ones(D)  # setup gsm with initilization from LBFGS fit
    mean_init, cov_init, lbfgs_res = lbfgs_init(mean_init, lp, lp_g)
    print(f'LBFGS fit: \n{lbfgs_res}\n')

    mean_gsm, cov_gsm, monitor_gsm = gsm_fit(D, lp, lp_g, mean_init, cov_init, lbfgs_res, niter, batch_size)
    mean_advi, cov_advi, monitor_advi = advi_fit(D, lp, lp_g, mean_init, cov_init, lbfgs_res, niter, batch_size)
    mean_bam2, cov_bam2, monitor_bam2 = bam_fit2(D, lp, lp_g, mean_init, cov_init, lbfgs_res, niter, batch_size)

    # Check that the output is correct
    print("\nTrue mean : ", mean)
    print("Fit gsm  : ", mean_gsm)
    print("Fit advi  : ", mean_advi)
    print("Fit bam2 : ", mean_bam2)

    print("Check mean fit2")
    print(np.allclose(mean, mean_bam2))
    print("Check cov fit2")
    print(np.allclose(cov, cov_bam2))

    # Check that the KL divergence decreases
    plt.plot(monitor_gsm.nevals, monitor_gsm.rkl, label='GSM')
    plt.plot(monitor_advi.nevals, monitor_advi.rkl, label='ADVI')
    plt.plot(monitor_bam2.nevals, monitor_bam2.rkl, label=f'BaM')

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Reverse KL")

    plt.yscale('log')
    plt.xscale('log')

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    picname = f"monitor_kl_{timestamp}.png"
    plt.savefig(picname)
    plt.show(block=True)
