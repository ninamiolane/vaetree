"""
Routines for forward and reverse AIS.
The sequence of distributions is given in terms of a Problem class,
which should provide the following methods:
  - init_sample():
    return an exact sample from the initial distribution
  - init_partition_function():
    compute the exact log partition function of the initial distribution
  - step(state, t):
    Given the current state, take an MCMC step with inverse temperature t,
    and return the new state
  - joint_prob(state, t):
    Return the joint unnormalized probability of a state,
    with inverse temperature t.

The Problem class for the VAE is implemented.
"""

import time

import numpy as np
import pdb

import imnn

nax = np.newaxis
DEBUGGER = None


class ProblemVAE(object):
    def __init__(self, vae, batch_data):
        self.vae = vae
        self.encoder = vae.encoder
        self.batch_data = batch_data

    def init_sample(self):
        """
        Return an exact sample from the initial distribution,
        q(z|x) in the case of the VAE.
        """
        muz, logvarz = self.encoder(batch_data)
        z_from_q = imnn.sample_from_q(muz, logvarz, n_samples=1)
        return z_from_q

    def init_partition_function(self):
        """
        Compute the exact *log* partition function
        of the initial distribution q(z|x).
        """

    def step(self, state, t):

    def joint_prob(self, state, t):




def sigmoid_schedule(num, rad=4):
    """The sigmoid schedule defined in Section 6.2 of the paper. This is defined as:
          gamma_t = sigma(rad * (2t/T - 1))
          beta_t = (gamma_t - gamma_1) / (gamma_T - gamma_1),
    where sigma is the logistic sigmoid. This schedule allocates more distributions near
    the inverse temperature 0 and 1, since these are often the places where the distributon
    changes the fastest.
    """
    if num == 1:
        return [np.asarray(0.0),np.asarray(1.0)]
    t = np.linspace(-rad, rad, num)
    sigm = 1. / (1. + np.exp(-t))
    return (sigm - sigm.min()) / (sigm.max() - sigm.min())

def LogMeanExp(A,axis=None):
    A_max = np.max(A, axis=axis, keepdims=True)
    B = (
        np.log(np.mean(np.exp(A - A_max), axis=axis, keepdims=True)) +
        A_max
    )
    return B



def ais(problem, schedule,sigma):
    """Run AIS in the forward direction. Problem is as defined above, and schedule should
    be an array of monotonically increasing values with schedule[0] == 0 and schedule[-1] == 1."""
    pf = problem.init_partition_function()
    deltas = []
    new = []
    prev = []
    index =1
    monitor = False
    for it, (t0, t1) in enumerate(zip(schedule[:-1], schedule[1:])):
        new_U = problem.eval_lld(t1.astype(np.float32))
        prev_U = problem.eval_lld(t0.astype(np.float32))
        delta = new_U - prev_U
        pf += delta
        obs_lld = LogMeanExp(pf,axis=0)
        obs_mean = np.mean(obs_lld)
        deltas.append(obs_mean)

        accept = problem.step(t1.astype(np.float32))
        if (index+1)%100 == 0:
            print ("\nsteps %i"%index)
            print ("Accept: "+str(np.mean(accept)))
            print ("Log-prob: " + str(np.mean(obs_mean)))
        index +=1
    finalstate = problem.h.get_value()
    obs_lld = LogMeanExp(pf,axis=0)
    obs_mean = np.mean(obs_lld)
    print ("Final Log-prob: " + str(obs_mean))
    return obs_mean, pf, finalstate


def reverse_ais(problem, schedule, sigma):
    """Run AIS in the reverse direction. Problem is as defined above, and schedule should
    be an array of monotonically increasing values with schedule[0] == 0 and schedule[-1] == 1.
    Init_state is an exact sample from the posterior_distribution."""
    pf_ratio = 0.
    deltas = []
    rev_schedule = schedule[::-1]
    index = 1
    for it, (t0, t1) in enumerate(zip(rev_schedule[:-1], rev_schedule[1:])):
        new_log = problem.eval_lld(t1.astype(np.float32))
        prev_log = problem.eval_lld(t0.astype(np.float32))
        delta = new_log - prev_log
        pf_ratio += delta
        obs_lld = LogMeanExp(-pf_ratio,axis=0)
        obs_mean = np.mean(obs_lld)
        deltas.append(obs_mean)

        accept = problem.step(t1.astype(np.float32))
        if (index+1)%100 == 0:
            print ("\nsteps %i"%index)
            print ("Accept: "+str(np.mean(accept)))
            print ("Log-prob: " + str(obs_mean))
        index+=1

    finalstate = problem.h.get_value()
    pf_estimate = problem.init_partition_function() - pf_ratio
    obs_lld = LogMeanExp(pf_estimate,axis=0)
    obs_mean = np.mean(obs_lld)
    print ("Final Log-prob: " + str(obs_mean))
    return obs_mean, pf_estimate, finalstate
