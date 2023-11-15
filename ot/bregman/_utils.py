# -*- coding: utf-8 -*-
"""
Common tools of Bregman projections solvers for entropic regularized OT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

from ..utils import list_to_array
from ..backend import get_backend


def geometricBar(weights, alldistribT):
    """return the weighted geometric mean of distributions"""
    weights, alldistribT = list_to_array(weights, alldistribT)
    nx = get_backend(weights, alldistribT)
    assert (len(weights) == alldistribT.shape[1])
    return nx.exp(nx.dot(nx.log(alldistribT), weights.T))


def geometricMean(alldistribT):
    """return the  geometric mean of distributions"""
    alldistribT = list_to_array(alldistribT)
    nx = get_backend(alldistribT)
    return nx.exp(nx.mean(nx.log(alldistribT), axis=1))


def projR(gamma, p):
    """return the KL projection on the row constraints """
    gamma, p = list_to_array(gamma, p)
    nx = get_backend(gamma, p)
    return (gamma.T * p / nx.maximum(nx.sum(gamma, axis=1), 1e-10)).T


def projC(gamma, q):
    """return the KL projection on the column constraints """
    gamma, q = list_to_array(gamma, q)
    nx = get_backend(gamma, q)
    return gamma * q / nx.maximum(nx.sum(gamma, axis=0), 1e-10)
