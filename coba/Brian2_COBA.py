import time

from brian2 import *
from scipy.io import mmread

def configure_and_run_brian2(simtime_in_s, num_threads):
    # ###########################################
    # Configuration
    # ###########################################
    numpy.random.seed(98765)
    set_device('cpp_standalone')
    prefs.devices.cpp_standalone.openmp_threads = num_threads

    # ###########################################
    # Network parameters
    # ###########################################
    taum = 20*ms
    taue = 5*ms
    taui = 10*ms
    Vt = -50*mV
    Vr = -60*mV
    El = -60*mV
    Erev_exc = 0.*mV
    Erev_inh = -80.*mV
    I = 20. * mvolt

    # ###########################################
    # Neuron model
    # ###########################################
    eqs = '''
    dv/dt  = (ge*(Erev_exc-v)+gi*(Erev_inh-v)-(v-El) + I)*(1./taum) : volt (unless refractory)
    dge/dt = -ge/taue : 1 
    dgi/dt = -gi/taui : 1 
    '''

    # ###########################################
    # Population
    # ###########################################
    NE = 3200
    NI = NE/4
    P = NeuronGroup(NE+NI, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms, method='euler')
    P.v = (randn(len(P)) * 5. - 55.) * mvolt
    Pe = P[:NE]
    Pi = P[NE:]

    # ###########################################
    # Projections
    # ###########################################
    we = 0.6 # excitatory synaptic weight (voltage)
    wi = 6.7 # inhibitory synaptic weight
    conn_ee = Synapses(Pe,Pe,model="w:1",on_pre='ge += w', method='euler')
    conn_ei = Synapses(Pe,Pi,model="w:1",on_pre='ge += w', method='euler')
    conn_ie = Synapses(Pi,Pe,model="w:1",on_pre='gi += w', method='euler')
    conn_ii = Synapses(Pi,Pi,model="w:1",on_pre='gi += w', method='euler')

    ee_mat = mmread('ee.wmat')
    ei_mat = mmread('ei.wmat')
    ie_mat = mmread('ie.wmat')
    ii_mat = mmread('ii.wmat')

    conn_ee.connect(i=ee_mat.row, j=ee_mat.col)
    conn_ee.w=we

    conn_ei.connect(i=ei_mat.row, j=ei_mat.col)
    conn_ei.w=we

    conn_ie.connect(i=ie_mat.row, j=ie_mat.col)
    conn_ie.w=wi

    conn_ii.connect(i=ii_mat.row, j=ii_mat.col)
    conn_ii.w=wi



    # ###########################################
    # Simulation
    # ###########################################
    s_mon = SpikeMonitor(P)
    # Run for 0 second in order to measure compilation time
    run(simtime_in_s * second, profile=True)
    totaltime = device._last_run_time

    print('Done in', totaltime)

    # ###########################################
    # Data analysis
    # ###########################################
    #plot(s_mon.t/ms, s_mon.i, '.')
    #xlabel('Time (ms)')
    #ylabel('Neuron index')
    #show()

    return totaltime

if __name__== "__main__":
    "stand alone run"
    nb_threads = 1
    if len(sys.argv) > 1:
        nb_threads = int(sys.argv[1])
    
    configure_and_run_brian2(10, nb_threads)
