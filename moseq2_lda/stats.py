from scipy import stats
import numpy as np


def compute_kde(data, mask, a1, a2, extents, res=100j):
    if extents is None:
        xx, yy = np.mgrid[np.min(data.T[a1]) : np.max(data.T[a1]) : res, np.min(data.T[a2]) : np.max(data.T[a2]) : res]
    else:
        xx, yy = np.mgrid[extents[a1][0] : extents[a1][1] : res, extents[a2][0] : extents[a2][1] : res]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([data[mask, a1], data[mask, a2]])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    return xx, yy, f


def compute_3D_kde(data, mask, extents, res=100j):
    if extents is None:
        xx, yy, zz = np.mgrid[
            np.min(data.T[0]) : np.max(data.T[0]) : res,
            np.min(data.T[1]) : np.max(data.T[1]) : res,
            np.min(data.T[2]) : np.max(data.T[2]) : res,
        ]
    else:
        xx, yy, zz = np.mgrid[extents[0][0] : extents[0][1] : res, extents[1][0] : extents[1][1] : res, extents[2][0] : extents[2][1] : res]

    positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
    values = np.vstack([data[mask, 0], data[mask, 1], data[mask, 2]])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    return xx, yy, zz, f
