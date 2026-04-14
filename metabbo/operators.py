import copy
import numpy as np
from typing import Iterable, Union


def clipping(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float, None],
             ub: Union[np.ndarray, Iterable, int, float, None]
             ) -> np.ndarray:
    return np.clip(x, lb, ub)


def random(x: Union[np.ndarray, Iterable],
           lb: Union[np.ndarray, Iterable, int, float],
           ub: Union[np.ndarray, Iterable, int, float]
           ) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(ub, np.ndarray):
        ub = np.array(ub)
    cro_bnd = (x < lb) | (x > ub)
    return ~cro_bnd * x + cro_bnd * (np.random.rand(*x.shape) * (ub - lb) + lb)


def reflection(x: Union[np.ndarray, Iterable],
               lb: Union[np.ndarray, Iterable, int, float],
               ub: Union[np.ndarray, Iterable, int, float]
               ) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    cro_lb = x < lb
    cro_ub = x > ub
    no_cro = ~(cro_lb | cro_ub)
    return no_cro * x + cro_lb * (2 * lb - x) + cro_ub * (2 * ub - x)


def periodic(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float],
             ub: Union[np.ndarray, Iterable, int, float]
             ) -> np.ndarray:
    if not isinstance(ub, np.ndarray):
        ub = np.array(ub)
    return (x - ub) % (ub - lb) + lb


def halving(x: Union[np.ndarray, Iterable],
            lb: Union[np.ndarray, Iterable, int, float],
            ub: Union[np.ndarray, Iterable, int, float]
            ) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    cro_lb = x < lb
    cro_ub = x > ub
    no_cro = ~(cro_lb | cro_ub)
    return no_cro * x + cro_lb * (x + lb) / 2 + cro_ub * (x + ub) / 2


def parent(x: Union[np.ndarray, Iterable],
           lb: Union[np.ndarray, Iterable, int, float],
           ub: Union[np.ndarray, Iterable, int, float],
           par: Union[np.ndarray, Iterable]
           ) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(par, np.ndarray):
        par = np.array(par)
    cro_lb = x < lb
    cro_ub = x > ub
    no_cro = ~(cro_lb | cro_ub)
    return no_cro * x + cro_lb * (par + lb) / 2 + cro_ub * (par + ub) / 2


def binomial(x: np.ndarray, v: np.ndarray, Cr: Union[np.ndarray, float]) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
        v = v.reshape(1, -1)
    NP, dim = x.shape
    jrand = np.random.randint(dim, size=NP)
    if isinstance(Cr, np.ndarray) and Cr.ndim == 1:
        Cr = Cr.reshape(-1, 1)
    u = np.where(np.random.rand(NP, dim) < Cr, v, x)
    u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
    if u.shape[0] == 1:
        u = u.squeeze(axis=0)
    return u


def exponential(x: np.ndarray, v: np.ndarray, Cr: Union[np.ndarray, float]) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
        v = v.reshape(1, -1)
    NP, dim = x.shape
    u = copy.deepcopy(x)
    L = np.random.randint(dim, size=(NP, 1)).repeat(dim).reshape(NP, dim)
    R = np.ones(NP) * dim
    rvs = np.random.rand(NP, dim)
    i = np.arange(dim).repeat(NP).reshape(dim, NP).transpose()
    if isinstance(Cr, np.ndarray) and Cr.ndim == 1:
        Cr = Cr.reshape(-1, 1)
    rvs[rvs > Cr] = np.inf
    rvs[i <= L] = -np.inf
    k = np.where(rvs == np.inf)
    ki = np.stack(k).transpose()
    if ki.shape[0] > 0:
        k_ = np.concatenate((ki, ki[None, -1] + 1), 0)
        _k = np.concatenate((ki[None, 0] - 1, ki), 0)
        ind = ki[(k_[:, 0] != _k[:, 0]).reshape(-1, 1).repeat(2).reshape(-1, 2)[:-1]].reshape(-1, 2).transpose()
        R[ind[0]] = ind[1]
    R = R.repeat(dim).reshape(NP, dim)
    u[(i >= L) * (i < R)] = v[(i >= L) * (i < R)]
    if u.shape[0] == 1:
        u = u.squeeze(axis=0)
    return u


def generate_random_int_single(NP: int, cols: int, pointer: int) -> np.ndarray:
    r = np.random.randint(low=0, high=NP, size=cols)
    while pointer in r:
        r = np.random.randint(low=0, high=NP, size=cols)
    return r


def generate_random_int(NP: int, cols: int) -> np.ndarray:
    r = np.random.randint(low=0, high=NP, size=(NP, cols))
    for col in range(0, cols):
        while True:
            is_repeated = [np.equal(r[:, col], r[:, i]) for i in range(col)]
            is_repeated.append(np.equal(r[:, col], np.arange(NP)))
            repeated_index = np.nonzero(np.any(np.stack(is_repeated), axis=0))[0]
            repeated_sum = repeated_index.size
            if repeated_sum != 0:
                r[repeated_index[:], col] = np.random.randint(low=0, high=NP, size=repeated_sum)
            else:
                break
    return r


def rand_1_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer)
    return x[r[0]] + F * (x[r[1]] - x[r[2]])


def rand_1(x: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 3)
    return x[r[:, 0]] + F * (x[r[:, 1]] - x[r[:, 2]])


def rand_2_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 5, pointer)
    return x[r[0]] + F * (x[r[1]] - x[r[2]] + x[r[3]] - x[r[4]])


def rand_2(x: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 5)
    return x[r[:, 0]] + F * (x[r[:, 1]] - x[r[:, 2]] + x[r[:, 3]] - x[r[:, 4]])


def best_1_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 2, pointer)
    return best + F * (x[r[0]] - x[r[1]])


def best_1(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 2)
    return best + F * (x[r[:, 0]] - x[r[:, 1]])


def best_2_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 4, pointer)
    return best + F * (x[r[0]] - x[r[1]] + x[r[2]] - x[r[3]])


def best_2(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 4)
    return best + F * (x[r[:, 0]] - x[r[:, 1]] + x[r[:, 2]] - x[r[:, 3]])


def rand_to_best_1_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer)
    return x[r[0]] + F * (best - x[r[0]] + x[r[1]] - x[r[2]])


def rand_to_best_1(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 3)
    return x[r[:, 0]] + F * (best - x[r[:, 0]] + x[r[:, 1]] - x[r[:, 2]])


def rand_to_best_2_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 5, pointer)
    return x[r[0]] + F * (best - x[r[0]] + x[r[1]] - x[r[2]] + x[r[3]] - x[r[4]])


def rand_to_best_2(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 5)
    return x[r[:, 0]] + F * (best - x[r[:, 0]] + x[r[:, 1]] - x[r[:, 2]] + x[r[:, 3]] - x[r[:, 4]])


def cur_to_best_1_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 2, pointer)
    return x[pointer] + F * (best - x[pointer] + x[r[0]] - x[r[1]])


def cur_to_best_1(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 2)
    return x + F * (best - x + x[r[:, 0]] - x[r[:, 1]])


def cur_to_best_2_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 4, pointer)
    return x[pointer] + F * (best - x[pointer] + x[r[0]] - x[r[1]] + x[r[2]] - x[r[3]])


def cur_to_best_2(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 4)
    return x + F * (best - x + x[r[:, 0]] - x[r[:, 1]] + x[r[:, 2]] - x[r[:, 3]])


def cur_to_rand_1_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer)
    return x[pointer] + F * (x[r[0]] - x[pointer] + x[r[1]] - x[r[2]])


def cur_to_rand_1(x: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 3)
    return x + F * (x[r[:, 0]] - x + x[r[:, 1]] - x[r[:, 2]])


def cur_to_rand_2_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 5, pointer)
    return x[pointer] + F * (x[r[0]] - x[pointer] + x[r[1]] - x[r[2]] + x[r[3]] - x[r[4]])


def cur_to_rand_2(x: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 5)
    return x + F * (x[r[:, 0]] - x + x[r[:, 1]] - x[r[:, 2]] - x[r[:, 3]] + x[r[:, 4]])
