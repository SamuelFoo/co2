import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter


def ukf_rts_smooth(t, y, R=1000, x0=None):
    sigmas = MerweScaledSigmaPoints(n=1, alpha=1e-3, beta=2, kappa=0.0)
    ukf = UnscentedKalmanFilter(
        dim_x=1,
        dim_z=1,
        dt=np.gradient(t).mean(),
        hx=lambda x: x,
        fx=lambda x, dt: x,
        points=sigmas,
    )

    if x0 is None:
        ukf.x = np.array([y[0]])
    else:
        ukf.x = np.array([x0])

    ukf.R *= R

    mu, cov = ukf.batch_filter(zs=y, dts=np.gradient(t))
    xs, Ps, Ks = ukf.rts_smoother(mu, cov)
    return xs
