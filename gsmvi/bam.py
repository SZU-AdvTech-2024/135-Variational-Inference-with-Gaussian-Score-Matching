import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy.linalg import sqrtm as sqrtm_jsp
from scipy.linalg import sqrtm as sqrtm_sp
import numpy as np
# import scipy.sparse as spys
from scipy.sparse.linalg import svds
from jax.lib import xla_bridge


# 用于低秩更新
def compute_Q_host(U_B):
    U, B = U_B
    # UU, DD, VV = spys.linalg.svds(U, k=B)
    U = np.array(U)
    UU, DD, VV = svds(U, k=B)  # 奇异值分解→三个矩阵
    # B：要计算的奇异值的个数，设为batchsize
    return UU * np.sqrt(DD)


def compute_Q(U_B):
    result_shape = jax.ShapeDtypeStruct((U_B[0].shape[0], U_B[1]), U_B[0].dtype)
    return jax.pure_callback(compute_Q_host, result_shape, U_B)
    # 将 compute_Q_host 函数包装为一个可以在 JAX 计算图中使用的纯函数


# 计算矩阵 M 的平方根
def get_sqrt(M):
    if xla_bridge.get_backend().platform == 'gpu':
        result_shape = jax.ShapeDtypeStruct(M.shape, M.dtype)
        M_root = jax.pure_callback(lambda x: sqrtm_sp(x).astype(M.dtype), result_shape, M)
        # sqrt can be complex sometimes, we only want real part
    elif xla_bridge.get_backend().platform == 'cpu':
        M_root = sqrtm_jsp(M)
    else:
        print("Backend not recongnized in get_sqrt function. Should be either gpu or cpu")
        raise
    return M_root.real


# 更新均值和协方差矩阵
def bam_update(samples, vs, mu0, S0, reg):
    """
    Returns updated mean and covariance matrix with batch and match updates.
    For a batch, this is simply the mean of updates for individual samples.

    Inputs:
      samples: Array of samples of shape BxD where B is the batch dimension
      vs : Array of score functions of shape BxD corresponding to samples
      得分
      mu0 : Array of shape D, current estimate of the mean
      S0 : Array of shape DxD, current estimate of the covariance matrix

    Returns:
      mu : Array of shape D, new estimate of the mean
      S : Array of shape DxD, new estimate of the covariance matrix
    """

    # (B, D)
    assert len(samples.shape) == 2
    assert len(vs.shape) == 2

    B = samples.shape[0]
    outer_map = jax.vmap(jnp.outer, in_axes=(0, 0))
    # 使用 jax.vmap 对 jnp.outer 进行向量化 → 可以对多个向量进行外积运算

    # 样本的协方差矩阵
    xbar = jnp.mean(samples, axis=0)
    xdiff = samples - xbar  # 每个样本与均值的差
    C = jnp.mean(outer_map(xdiff, xdiff), axis=0)

    # 得分函数的协方差矩阵
    gbar = jnp.mean(vs, axis=0)
    gdiff = vs - gbar
    G = jnp.mean(outer_map(gdiff, gdiff), axis=0)

    # 3.1 Match Step
    U = reg * G + reg / (1 + reg) * jnp.outer(gbar, gbar)
    V = S0 + reg * C + reg / (1 + reg) * jnp.outer(mu0 - xbar, mu0 - xbar)
    I = jnp.identity(samples.shape[1])
    # DxD的单位矩阵

    mat = I + 4 * jnp.matmul(U, V)
    S = 2 * jnp.matmul(V, jnp.linalg.inv(I + get_sqrt(mat)))
    # S = 2 * jnp.linalg.solve(I + get_sqrt(mat).T, V.T)
    mu = 1 / (1 + reg) * mu0 + reg / (1 + reg) * (jnp.matmul(S, gbar) + xbar)

    return mu, S


# 低秩BaM更新均值和协方差矩阵
def bam_lowrank_update(samples, vs, mu0, S0, reg):
    """
    Returns updated mean and covariance matrix with low-rank BaM updates.
    For a batch, this is simply the mean of updates for individual samples.

    Inputs:
      samples: Array of samples of shape BxD where B is the batch dimension
      vs : Array of score functions of shape BxD corresponding to samples
      mu0 : Array of shape D, current estimate of the mean
      S0 : Array of shape DxD, current estimate of the covariance matrix

    Returns:
      mu : Array of shape D, new estimate of the mean
      S : Array of shape DxD, new estimate of the covariance matrix
    """

    assert len(samples.shape) == 2
    assert len(vs.shape) == 2
    B = samples.shape[0]
    xbar = jnp.mean(samples, axis=0)
    outer_map = jax.vmap(jnp.outer, in_axes=(0, 0))
    xdiff = samples - xbar
    C = jnp.mean(outer_map(xdiff, xdiff), axis=0)

    gbar = jnp.mean(vs, axis=0)
    gdiff = vs - gbar
    G = jnp.mean(outer_map(gdiff, gdiff), axis=0)

    U = reg * G + (reg) / (1 + reg) * jnp.outer(gbar, gbar)
    V = S0 + reg * C + (reg) / (1 + reg) * jnp.outer(mu0 - xbar, mu0 - xbar)

    # Form decomposition that is D x K
    # 低秩分解
    Q = compute_Q((U, B))  # 低秩矩阵Q，用于近似矩阵U
    I = jnp.identity(B)  # 大小为B的单位矩阵
    VT = V.T  # 转置
    A = VT.dot(Q)
    BB = 0.5 * I + jnp.real(get_sqrt(A.T.dot(Q) + 0.25 * I))
    BB = BB.dot(BB)
    CC = jnp.linalg.solve(BB, A.T)
    S = VT - A @ CC
    mu = 1 / (1 + reg) * mu0 + reg / (1 + reg) * (jnp.matmul(S, gbar) + xbar)

    return mu, S


class BaM:
    """
    Wrapper class for using BaM updates to fit a distribution
    """

    def __init__(self, D, lp, lp_g, use_lowrank=False, jit_compile=True):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters
          lp : Function to evaluate target log-probability distribution.
                         (Only used in monitor, not for fitting)
          lp_g : Function to evaluate score, i.e. the gradient of the target log-probability distribution

          是否使用低秩近似（减少计算成本和内存使用）和 JIT 编译（加速计算）
        """

        self.D = D
        self.lp = lp
        self.lp_g = lp_g
        self.use_lowrank = use_lowrank
        if use_lowrank:
            print("Using lowrank update")
        self.jit_compile = jit_compile
        if not jit_compile:
            print("Not using jit compilation. This may take longer than it needs to.")

    def fit(self, key, regf, mean=None, cov=None, batch_size=2, niter=5000, nprint=10, verbose=True,
            check_goodness=True, monitor=None, retries=10, jitter=1e-6):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Inputs:
          key: Random number generator key (jax.random.PRNGKey)
          regf : Function to return regularizer value at an iteration. See Regularizers class below
          正则化函数
          mean : Optional, initial value of the mean. Expected None or array of size D
          cov : Optional, initial value of the covariance matrix. Expected None or array of size DxD
          batch_size : Optional, int. Number of samples to match scores for at every iteration
          niter : Optional, int. Total number of iterations
          最大迭代次数
          nprint : Optional, int. Number of iterations after which to print logs
          verbose : Optional, bool. If true, print number of iterations after nprint
          check_goodness : Optional, bool. Recommended. Wether to check floating point errors in covariance matrix update
          monitor : Optional. Function to monitor the progress and track different statistics for diagnostics.
                    Function call should take the input tuple (iteration number, [mean, cov], lp, key, number of grad evals).
                    Example of monitor class is provided in utils/monitors.py

        Returns:
          mu : Array of shape D, fit of the mean
          cov : Array of shape DxD, fit of the covariance matrix
        """
        # 初始化
        if mean is None:
            mean = jnp.zeros(self.D)
        if cov is None:
            cov = jnp.identity(self.D)  # DxD

        nevals = 1  # 梯度评估次数

        if self.use_lowrank:
            update_function = bam_lowrank_update
        else:
            update_function = bam_update
        if self.jit_compile:
            update_function = jit(update_function)

        if nprint > niter:
            nprint = niter

        for i in range(niter + 1):
            if (i % (niter // nprint) == 0) and verbose:
                print(f'Iteration {i} of {niter}')

            # 调用监控函数 monitor，并将梯度评估次数 nevals 重置为 0
            if monitor is not None:
                if (i % monitor.checkpoint) == 0:
                    monitor(i, [mean, cov], self.lp, key, nevals=nevals)
                    nevals = 0

            # Can generate samples from jax distribution (commented below), but using numpy is faster
            j = 0
            while True:  # Sometimes run crashes due to a bad sample. Avoid that by re-trying.
                try:
                    key, key_sample = random.split(key, 2)
                    np.random.seed(key_sample[0])
                    # 从当前高斯分布中抽取 batch_size 个样本
                    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=batch_size)
                    # 计算样本的得分
                    vs = self.lp_g(samples)
                    nevals += batch_size
                    # 正则化
                    reg = regf(i)
                    # 更新参数
                    mean_new, cov_new = update_function(samples, vs, mean, cov, reg)
                    # 向新的协方差矩阵添加一个小抖动，确保稳定性
                    cov_new += np.eye(self.D) * jitter  # jitter covariance matrix
                    # 确保新的协方差矩阵仍是对称的→浮点精度限制和算法的近似
                    cov_new = (cov_new + cov_new.T) / 2.
                    break

                except Exception as e:
                    if j < retries:
                        j += 1
                        print(f"Failed with exception {e}")
                        print(f"Trying again {j} of {retries}")
                    else:
                        raise e

            # 检查新的协方差矩阵是否有效
            is_good = self._check_goodness(cov_new)
            if is_good:
                mean, cov = mean_new, cov_new
                # print("Good update for covariance matrix.")
            else:
                if verbose:
                    print("Bad update for covariance matrix. Revert")

        if monitor is not None:
            monitor(i, [mean, cov], self.lp, key, nevals=nevals)
        return mean, cov

    # 检查协方差矩阵是否有效（正定）
    def _check_goodness(self, cov):
        '''
        Internal function to check if the new covariance matrix is a valid covariance matrix.
        Required due to floating point errors in updating the convariance matrix directly,
        insteead of it's Cholesky form.
        '''
        is_good = False
        try:
            if (np.isnan(np.linalg.cholesky(cov))).any():
                # nan_update.append(j)
                print("Cholesky 分解结果中有 NaN 值")
            else:
                is_good = True
            return is_good
        except:
            print("Cholesky 分解失败")
            return is_good


class Regularizers():
    """
    Class for regularizers used in BaM
    """

    def __init__(self):
        self.counter = 0  # 被调用次数

    def reset(self):
        self.counter = 0

    def constant(self, reg0):
        def reg_iter(iteration):
            self.counter += 1
            return reg0  # 返回常数正则化值

        return reg_iter

    def linear(self, reg0):
        def reg_iter(iteration):
            self.counter += 1
            return reg0 / self.counter  # 返回线性递减正则化值

        return reg_iter

    def custom(self, func):
        def reg_iter(iteration):
            self.counter += 1
            return func(self.counter)  # 返回自定义正则化函数

        return reg_iter
