import numpy as np
from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
from jax import jit, grad, random
from numpyro.distributions import MultivariateNormal

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


# 反向 KL 散度， 1/n ∑(log q - log p)
def reverse_kl(samples, lpq, lpp):
    logl = np.sum(lpp(samples))
    logq = np.sum(lpq(samples))
    rkl = logq - logl
    if rkl < 0:
        rkl *= -1
    rkl /= samples.shape[0]
    return rkl


# 正向 KL 散度， 1/n ∑(log p - log q)
def forward_kl(samples, lpq, lpp):
    logl = np.sum(lpp(samples))
    logq = np.sum(lpq(samples))
    fkl = logl - logq
    fkl /= samples.shape[0]
    return fkl


# JIT 优化，第3个参数 lp 是静态的
@partial(jit, static_argnums=(3))
def reverse_kl_jit(samples, mu, cov, lp):
    q = MultivariateNormal(mu, cov)
    # 多元正态分布对象 q
    logq = jnp.sum(q.log_prob(samples))
    logl = jnp.sum(lp(samples))
    rkl = logq - logl
    if rkl < 0:
        rkl *= -1
    rkl /= samples.shape[0]
    return rkl


@partial(jit, static_argnums=(3))
def forward_kl_jit(samples, mu, cov, lp):
    q = MultivariateNormal(mu, cov)
    logq = jnp.sum(q.log_prob(samples))
    logl = jnp.sum(lp(samples))
    fkl = logl - logq
    fkl /= samples.shape[0]
    return fkl


@dataclass
class KLMonitor():
    """
    Class to monitor KL divergence during optimization for VI
    
    Inputs:
    
    batch_size_kl: (int) Number of samples to use to estimate KL divergence
    checkpoint: (int) Number of iterations after which to run monitor
    offset_evals: (int) Value with which to offset number of gradient evaluatoins
                    Used to account for gradient evaluations done in warmup or initilization        
    ref_samples: Optional, samples from the target distribution.
                   If provided, also track forward KL divergence
    """

    # batch_size_kl: int = 8
    # checkpoint: int = 20
    # # 每隔几次迭代运行监控器
    # offset_evals: int = 0
    # # 调整梯度评估次数的偏移量，用于预训练或初始化等
    # ref_samples: np.array = None
    # 提供：可跟踪前向 KL 散度

    def __init__(self, batch_size_kl=8, checkpoint=20, offset_evals=0, ref_samples=None):

        self.rkl = []
        self.fkl = []
        self.nevals = []
        # 新评估次数
        self.batch_size_kl = batch_size_kl
        self.checkpoint = checkpoint
        self.offset_evals = offset_evals
        self.ref_samples = ref_samples

    def reset(self,
              batch_size_kl=None,
              checkpoint=None,
              offset_evals=None,
              ref_samples=None):
        self.nevals = []
        self.rkl = []
        self.fkl = []
        if batch_size_kl is not None: self.batch_size_kl = batch_size_kl
        if checkpoint is not None: self.checkpoint = checkpoint
        if offset_evals is not None: self.offset_evals = offset_evals
        if ref_samples is not None: self.ref_samples = ref_samples
        print('offset evals reset to : ', self.offset_evals)

    def __call__(self, i, params, lp, key, nevals=1):
        """
        Main function to monitor reverse (and forward) KL divergence over iterations.

        Inputs:
        
        i: (int) iteration number
        params: (tuple; (mean, cov)) Current estimate of mean and covariance matrix
        lp: Function to evaluate target log-probability
        key: Random number generator key (jax.random.PRNGKey)
        nevals: (int) Number of gradient evaluations SINCE the last call of the monitor function
        自上次调用监控函数以来的梯度评估次数
        Returns:
        key : New key for generation random number
        """


        mu, cov = params
        key, key_sample = random.split(key)
        np.random.seed(key_sample[0])

        try:
            # 计算+储存反向 KL
            qsamples = np.random.multivariate_normal(mean=mu, cov=cov, size=self.batch_size_kl)
            q = MultivariateNormal(loc=mu, covariance_matrix=cov)
            self.rkl.append(reverse_kl(qsamples, q.log_prob, lp))
            # 计算+储存前向 KL
            if self.ref_samples is not None:
                idx = np.random.permutation(self.ref_samples.shape[0])[:self.batch_size_kl]
                psamples = self.ref_samples[idx]
                self.fkl.append(forward_kl(psamples, q.log_prob, lp))
            else:
                self.fkl.append(np.NaN)

        # 前向反向都没
        except Exception as e:
            print(f"Exception occured in monitor : {e}.\nAppending NaN")
            self.rkl.append(np.NaN)
            self.fkl.append(np.NaN)

        self.nevals.append(self.offset_evals + nevals)
        self.offset_evals = self.nevals[-1]

        return key
        # 返回分割后的新 key
