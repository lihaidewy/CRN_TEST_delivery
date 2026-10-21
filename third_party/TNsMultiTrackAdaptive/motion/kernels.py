"""运动模型数值内核（纯函数，单目标）。

- CTRV：来自 MATLAB ``D2Tracks.m`` 的 ``ctrv_transition`` / ``ctrv_predict`` / ``ctrv_update``
  （UKF，``alpha=1e-3, kappa=0, beta=2``）；
- CV：同一文件的线性卡尔曼形式（``F`` 由 ``dt`` 构造）。

与 MATLAB 的差别只有一处：MATLAB 为批量矩阵运算（``[dim x n]``），本设计**逐航迹调用**，
故内核为单目标签名（``x: (dim,)``、``P: (dim, dim)``），少一层 reshape/squeeze，语义不变。
"""

from __future__ import annotations

import numpy as np

# UKF 参数（与 MATLAB 一致）
ALPHA = 1e-3
BETA = 2.0
KAPPA = 0.0

STATE_DIM = 5          # CTRV
CV_STATE_DIM = 4       # CV


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """角度归一化到 ``[-pi, pi)``。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def ukf_weights(state_dim: int = STATE_DIM) -> tuple[np.ndarray, np.ndarray, float]:
    """返回 ``(Wm, Wc, sqrt_factor)``。"""
    lam = ALPHA**2 * (state_dim + KAPPA) - state_dim
    Wm = np.full(2 * state_dim + 1, 1.0 / (2.0 * (state_dim + lam)))
    Wm[0] = lam / (state_dim + lam)
    Wc = Wm.copy()
    Wc[0] += 1.0 - ALPHA**2 + BETA
    return Wm, Wc, float(np.sqrt(state_dim + lam))


def symmetrize(matrix: np.ndarray, regularizer: float = 1e-6) -> np.ndarray:
    """对称化并加正则项，保证后续 ``chol`` 可用。"""
    matrix = (matrix + matrix.T) / 2.0
    return matrix + regularizer * np.eye(matrix.shape[0])


def safe_cholesky(matrix: np.ndarray) -> np.ndarray:
    """``chol``，失败时用特征值分解兜底（与 MATLAB 一致）。"""
    try:
        return np.linalg.cholesky(symmetrize(matrix))
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(symmetrize(matrix))
        values = np.maximum(values, 1e-6)
        return np.linalg.cholesky(symmetrize(vectors @ np.diag(values) @ vectors.T))


def ctrv_transition(x: np.ndarray, dt: float) -> np.ndarray:
    """CTRV 状态转移 ``[x, y, v, psi, psi_dot]``（非线性，含 ``v>=0`` 约束）。"""
    x_pos, y_pos, v, psi, psi_dot = (float(x[i]) for i in range(STATE_DIM))
    if v < 0.0:
        v = -v
        psi = wrap_to_pi(psi + np.pi)

    if abs(v) < 1e-6:
        return np.array([x_pos, y_pos, v, wrap_to_pi(psi + psi_dot * dt), psi_dot])
    if abs(psi_dot) < 1e-6:
        return np.array(
            [x_pos + v * np.cos(psi) * dt, y_pos + v * np.sin(psi) * dt, v, wrap_to_pi(psi), psi_dot]
        )
    return np.array(
        [
            x_pos + (v / psi_dot) * (np.sin(psi + psi_dot * dt) - np.sin(psi)),
            y_pos + (v / psi_dot) * (-np.cos(psi + psi_dot * dt) + np.cos(psi)),
            v,
            wrap_to_pi(psi + psi_dot * dt),
            psi_dot,
        ]
    )


def _angle_aware(diff: np.ndarray) -> np.ndarray:
    """把航向分量（索引 3）的偏差归一化到 ``[-pi, pi)``。

    为什么必须做：``ctrv_transition`` 会对航向做 ``wrap_to_pi``，当航向跨过 ±π
    分支割线时，某个 Sigma 点的航向会从 −π 翻到 +π，原始偏差变成 ~2π。
    MATLAB 的 UKF 权重（``alpha=1e-3`` 时 ``W0≈−1e6``、``W_i≈1e5``）会把这个
    数值假象放大成 ~1e12 的方差，随后整条航迹发散到 1e6 量级。
    对角度取环绕偏差是 UKF 处理角度状态的标准做法（见设计报告差异表 D25）。
    """
    result = diff.copy()
    result[3] = wrap_to_pi(result[3])
    return result


def _weighted_mean(sigma: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """角度感知的 Sigma 点加权均值（以第 0 个 Sigma 点为参考）。"""
    reference = sigma[:, 0]
    difference = _angle_aware(sigma - reference[:, None])
    mean = reference + difference @ weights
    mean[3] = wrap_to_pi(mean[3])
    return mean


def ctrv_predict(
    x: np.ndarray, covariance: np.ndarray, q: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """UKF 一步预测：返回 ``(x_pred, P_pred)``。"""
    x = np.asarray(x, dtype=float).reshape(STATE_DIM)
    covariance = symmetrize(np.asarray(covariance, dtype=float))
    Wm, Wc, sqrt_factor = ukf_weights()
    sqrt_p = safe_cholesky(covariance)

    sigma = np.column_stack([x, x[:, None] + sqrt_factor * sqrt_p, x[:, None] - sqrt_factor * sqrt_p])
    sigma_pred = np.column_stack([ctrv_transition(sigma[:, i], dt) for i in range(sigma.shape[1])])

    x_pred = _weighted_mean(sigma_pred, Wm)
    diff = _angle_aware(sigma_pred - x_pred[:, None])
    p_pred = (diff * Wc) @ diff.T + q
    return x_pred, symmetrize(p_pred)


def ctrv_update(
    x_pred: np.ndarray,
    p_pred: np.ndarray,
    measurement: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """UKF 更新（量测线性：直接观测 x、y）：返回 ``(x_upd, P_upd)``。

    与 MATLAB 一致：用 Sigma 点统计量测预测与协方差，再算增益；
    角度归一化、``v>=0`` 约束、协方差对称化都在这里完成。
    """
    x_pred = np.asarray(x_pred, dtype=float).reshape(STATE_DIM)
    p_pred = symmetrize(np.asarray(p_pred, dtype=float))
    z = np.asarray(measurement, dtype=float).reshape(-1)
    Wm, Wc, sqrt_factor = ukf_weights()
    sqrt_p = safe_cholesky(p_pred)

    sigma = np.column_stack(
        [x_pred, x_pred[:, None] + sqrt_factor * sqrt_p, x_pred[:, None] - sqrt_factor * sqrt_p]
    )
    z_sigma = h @ sigma
    z_pred = z_sigma @ Wm

    diff_z = z_sigma - z_pred[:, None]
    s = (diff_z * Wc) @ diff_z.T + r
    s = symmetrize(s)

    diff_x = _angle_aware(sigma - x_pred[:, None])       # 航向按环绕偏差处理
    p_xz = (diff_x * Wc) @ diff_z.T

    gain = p_xz @ np.linalg.inv(s)
    x_upd = x_pred + gain @ (z - z_pred)
    x_upd[3] = wrap_to_pi(x_upd[3])
    if x_upd[2] < 0.0:                      # 速度约束：v>=0，航向转 pi
        x_upd[2] = -x_upd[2]
        x_upd[3] = wrap_to_pi(x_upd[3] + np.pi)

    p_upd = p_pred - gain @ s @ gain.T
    return x_upd, symmetrize(p_upd)


# ── CV（线性卡尔曼，状态 [x, y, vx, vy]）────────────────────────────────────
def cv_transition_matrix(dt: float) -> np.ndarray:
    """匀速模型状态转移矩阵（``dt`` 为实际时间差，不假设固定帧间隔）。"""
    matrix = np.eye(CV_STATE_DIM)
    matrix[0, 2] = dt
    matrix[1, 3] = dt
    return matrix


def cv_predict(
    x: np.ndarray, covariance: np.ndarray, q: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """一步预测：``x⁻ = F x``、``P⁻ = F P Fᵀ + Q``。"""
    x = np.asarray(x, dtype=float).reshape(CV_STATE_DIM)
    covariance = np.asarray(covariance, dtype=float)
    matrix = cv_transition_matrix(dt)
    return matrix @ x, symmetrize(matrix @ covariance @ matrix.T + q)


def cv_update(
    x_pred: np.ndarray,
    p_pred: np.ndarray,
    measurement: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """量测更新：标准卡尔曼（``K = P⁻Hᵀ(H P⁻ Hᵀ + R)⁻¹``）。"""
    x_pred = np.asarray(x_pred, dtype=float).reshape(CV_STATE_DIM)
    p_pred = np.asarray(p_pred, dtype=float)
    z = np.asarray(measurement, dtype=float).reshape(-1)

    innovation_covariance = h @ p_pred @ h.T + r
    gain = p_pred @ h.T @ np.linalg.inv(innovation_covariance)
    x_upd = x_pred + gain @ (z - h @ x_pred)
    p_upd = (np.eye(CV_STATE_DIM) - gain @ h) @ p_pred
    return x_upd, symmetrize(p_upd)
