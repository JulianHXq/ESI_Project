"""
core.py  --  shared kernel for the unified v5 project.  [v5-unify]

Single source of truth for the CES social-norm primitives used by BOTH parts of
the project (the Boucher et al. replication and the peers-of-peers assignment).
These were previously duplicated across the DGP and estimation modules; they are
consolidated here so there is exactly one implementation. The validation checks
are the union of the three former copies, so this is a behavior-preserving
drop-in for every caller.

    S_i(beta) = ( sum_j g_ij y_j^beta )^(1/beta)      CES social norm (0 if isolated)
    dS_i/dbeta                                          closed form (the beta-instrument)
    (G X)_i = sum_j g_ij x_j                            row-normalized peer average
"""
from __future__ import annotations
import numpy as np


def ces_norm(G, y, beta, isolated):
    """CES social norm S_i(beta) = (sum_j g_ij y_j^beta)^(1/beta); 0 for isolated.
    Assumes y strictly positive and beta>0 (we do NOT clip: clipping would change
    the object being estimated)."""
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("CES norm requires strictly positive outcomes. "
                         "Check the DGP or first-stage predicted outcomes.")
    if beta <= 0:
        raise ValueError("CES norm assumes beta > 0.")
    y_beta = y ** beta
    A = G @ y_beta
    S = np.zeros_like(y, dtype=float)
    active = ~isolated
    if np.any(A[active] <= 0):
        raise ValueError("CES norm is undefined for non-positive peer sums.")
    S[active] = A[active] ** (1.0 / beta)
    return S


def ces_norm_derivative(G, y, log_y, beta, isolated, S):
    """dS_i/dbeta in closed form (used to build the beta-instrument D = dS/dbeta):
        dS_i/dbeta = S_i * [ B_i/(beta A_i) - log(A_i)/beta^2 ],
    with A_i = sum_j g_ij y_j^beta and B_i = sum_j g_ij y_j^beta log(y_j)."""
    y = np.asarray(y, dtype=float)
    log_y = np.asarray(log_y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("CES derivative requires strictly positive outcomes.")
    if y.shape != log_y.shape:
        raise ValueError("y and log_y must have the same shape.")
    if beta <= 0:
        raise ValueError("beta must be > 0.")
    y_beta = y ** beta
    A = G @ y_beta
    B = G @ (y_beta * log_y)
    D = np.zeros_like(y, dtype=float)
    active = ~isolated
    D[active] = S[active] * (-np.log(A[active]) / (beta ** 2)
                            + B[active] / (beta * A[active]))
    return D


def peer_average(G, X):
    """Row-normalized peer average of a matrix of characteristics: (G X)."""
    return np.asarray(G @ X)
