Variational Inference (VI) with Gaussian Score Matching (GSM) (NeurIPS 2023)

GSM-VI, which is described in the NeurIPS 2023 paper https://arxiv.org/abs/2307.07849.



GSM-VI fits a multivariate Gasussian distribution with dense covaraince matrix to the target distribution
by score matching. It only requires access to the score function i.e. the gradient of the target log-probability
distribution and implements analytic updates for the variational parameters (mean and covariance matrix).



To install, run

```
pip install numpy==1.26.4
pip install opencv-python==4.10.0.84
pip install opencv-contrib-python==4.10.0.84
pip install jax==0.4.33
```


The example usage code is in `initializers.py`:

```
python3 initializers.py
```



