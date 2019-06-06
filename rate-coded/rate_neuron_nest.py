import nest
import pylab
import numpy
import time

dt = 0.1  # the resolution in ms
TrialLength = 100.0  # Simulation time in ms
NumTrials = 10

print("Building network")
def build_network(order, T=1):
    NE = int(4 * order)  # number of excitatory neurons
    NI = int(1 * order)  # number of inhibitory neurons
    N = int(NE+NI)       # total number of neurons
    
    d_e = 5.   # delay of excitatory connections in ms
    g = 5.0  # ratio inhibitory weight/excitatory weight
    epsilon = 0.1  # connection probability
    w = 0.1/numpy.sqrt(N)  # excitatory connection strength
    
    KE = int(epsilon * NE)  # number of excitatory synapses per neuron (outdegree)
    KI = int(epsilon * NI)  # number of inhibitory synapses per neuron (outdegree)
    K_tot = int(KI + KE)  # total number of synapses per neuron
    connection_rule = 'fixed_outdegree'  # connection rule
    print('number neurons: ', N)
    print('number synapses per neuron: ', K_tot)
    
    neuron_model = 'lin_rate_ipn'  # neuron model
    neuron_params = {'linear_summation': True,
                     # type of non-linearity (not affecting linear rate models)
                     'tau': 10.0,
                     # time constant of neuronal dynamics in ms
                     'mean': 0.0,
                     # mean of Gaussian white noise input
                     'std': 5.
                     # standard deviation of Gaussian white noise input
                     }

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt,
                          "use_wfr": False,
                          "local_num_threads": T,
                          #"print_time": True,
                          "overwrite_files": True})

    nest.SetDefaults(neuron_model, neuron_params)

    n_e = nest.Create(neuron_model, NE)
    n_i = nest.Create(neuron_model, NI)

    mm = nest.Create('multimeter', params={'record_from': ['rate'],
                                           'interval': dt})
    syn_e = {'weight': w, 'delay': d_e, 'model': 'rate_connection_delayed'}
    syn_i = {'weight': -g*w, 'model': 'rate_connection_instantaneous'}
    conn_e = {'rule': connection_rule, 'outdegree': KE}
    conn_i = {'rule': connection_rule, 'outdegree': KI}

    nest.Connect(n_e, n_e, conn_e, syn_e)
    nest.Connect(n_i, n_i, conn_i, syn_i)
    nest.Connect(n_e, n_i, conn_i, syn_e)
    nest.Connect(n_i, n_e, conn_e, syn_i)

    nest.Connect(mm, n_e+n_i)

    return n_e, n_i, mm

def measure():
    Order = [100, 200, 300, 400, 500, 600, 700, 800]

    avg_times = numpy.zeros((len(Order),NumTrials))

    for i, order in enumerate(Order):

        for j in range(NumTrials):
            n_e, n_i, mm = build_network(order)

            t1 = time.time()
            nest.Simulate(TrialLength)
            avg_times[i, j] = time.time()-t1

    return avg_times

def measure2():
    order = 800
    T = numpy.arange(1,6)

    avg_times = numpy.zeros((len(T),NumTrials))

    for i, t in enumerate(T):

        for j in range(NumTrials):
            n_e, n_i, mm = build_network(order, t)

            t1 = time.time()
            nest.Simulate(TrialLength)
            avg_times[i, j] = time.time()-t1

    return avg_times

if __name__ == "__main__":
    # Scaling across number of units (Figure 2)
    data = measure()
    numpy.savetxt("nest_times_st.csv", data, fmt="%.4f", delimiter=",")

    # Scaling across number of threads (Figure 3)
    data = measure2()
    numpy.savetxt("nest_times_omp.csv", data, fmt="%.4f", delimiter=",")

    # Validation
    n_e, n_i, mm = build_network(50)
    t1 = time.time()
    nest.Simulate(TrialLength)
    print('1:', time.time()-t1, 'seconds')
    data = nest.GetStatus(mm)[0]['events']
    rate_ex = data['rate'][numpy.where(data['senders'] == n_e[0])]
    rate_in = data['rate'][numpy.where(data['senders'] == n_i[0])]
    times = data['times'][numpy.where(data['senders'] == n_e[0])]

    pylab.figure()
    pylab.plot(times, rate_ex, label='excitatory')
    pylab.plot(times, rate_in, label='inhibitory')
    pylab.xlabel('time (ms)')
    pylab.ylabel('rate (a.u.)')
    pylab.legend()
    pylab.show()
