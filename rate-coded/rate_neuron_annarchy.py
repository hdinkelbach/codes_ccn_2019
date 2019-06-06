import numpy
import pylab
import os

from ANNarchy import *
setup(dt = 0.1)      # the resolution in ms

TrialLength = 100.0  # Simulation time in ms
NumTrials = 10

Lin_rate_ipn = Neuron(
    parameters = """
        tau = 10.0
        mean = 0.0 : population
        sigma = 5.0 : population
    """,
    equations = """
        r += dt/tau*(sum(exc) - sum(inh) - r) + sigma * Normal(mean, 1) / sqrt(tau)
    """
)

def build_network(order):
    clear()

    NE = int(4 * order)     # number of excitatory neurons
    NI = int(1 * order)     # number of inhibitory neurons
    N = int(NE+NI)          # total number of neurons

    d_e = 5.                # delay of excitatory connections in ms
    g = 5.0                 # ratio inhibitory weight/excitatory weight
    epsilon = 0.1           # connection probability
    w = 0.1/numpy.sqrt(N)   # excitatory connection strength

    KE = int(epsilon * NE)  # number of excitatory synapses per neuron (outdegree)
    KI = int(epsilon * NI)  # number of inhibitory synapses per neuron (outdegree)
    K_tot = int(KI + KE)    # total number of synapses per neuron

    # Network Config
    print('number neurons: ', N)
    print('number synapses per neuron: ', K_tot)

    # Create a population
    pop = Population(N, Lin_rate_ipn)

    # two views for pattern creation
    exc_pop = pop[:NE]
    inh_pop = pop[NE:]

    # Create projections
    ee = Projection(exc_pop, exc_pop, 'exc').connect_fixed_number_pre(KE, weights = w, delays=d_e)
    ii = Projection(inh_pop, inh_pop, 'inh').connect_fixed_number_pre(KI, weights = g*w)
    ei = Projection(exc_pop, inh_pop, 'exc').connect_fixed_number_pre(KI, weights = w, delays=d_e)
    ie = Projection(inh_pop, exc_pop, 'inh').connect_fixed_number_pre(KE, weights = g*w)

    compile(clean=True) # force recompile
    m = Monitor(pop, 'r')

    return exc_pop, inh_pop, m

def measure(Order=50):
    """
    Measure a set of benchmarks with varying number of neurons [500..4000].
    """
    avg_times = np.zeros(NumTrials)

    for i in range(NumTrials):
        n_e, n_i, mm = build_network(order)

        t1 = time.time()
        simulate(TrialLength)
        avg_times[i] = time.time()-t1

    return avg_times

if __name__ == "__main__":

    # Validation
    n_e, n_i, mm = build_network(50)
    simulate(TrialLength)
    data = mm.get('r')
    pylab.figure()
    # Plot the first exc and inh neuron
    pylab.plot(np.arange(0, len(data[:,0]))*dt(), data[:, n_e.ranks[0]])
    pylab.plot(np.arange(0, len(data[:,0]))*dt(), data[:, n_i.ranks[0]])
    pylab.show()
