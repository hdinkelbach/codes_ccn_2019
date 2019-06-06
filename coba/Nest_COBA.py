#
# An updated version of Nest-COBA implementation of Vitay et al. 2015 (doi: 10.3389/fninf.2015.00019).
#
# Uses the pyNEST interface of NEST 2.16 (doi: 10.5281/zenodo.1400175)
#
# Author: Helge Dinkelbach, Julien Vitay
#
from nest import *
import time
import numpy
from scipy.io import mmread

def configure_and_run_nest(simtime_in_s, num_threads):
    """
    Build up the network and run for a period *simtime_in_s*.
    The function returns the simulation time for 1 run without
    compilation time including time for record.
    """
    # ###########################################
    # Configuration
    # ###########################################
    numpy.random.seed(98765)

    ResetKernel()

    nest_kernel_config=    {
        "resolution": 0.1,
        "overwrite_files": True,
        "local_num_threads": num_threads
        # "rng_seeds": [98765] # works only for single threads otherwise I need different seeds for each thread
    }

    SetKernelStatus(nest_kernel_config)

    # ###########################################
    # Network parameters
    # ###########################################
    NE       =  3200   #number of exc. neurons
    NI       =   800   #number of inh. neurons

    # ###########################################
    # Neuron model
    # ###########################################
    SetDefaults("iaf_cond_exp", {
        "C_m"  : 200.,
        "g_L": 10.,
        "tau_syn_ex": 5.,
        "tau_syn_in": 10.,
        "E_ex": 0.,
        "E_in": -80.,
        "t_ref": 5.,
        "E_L"  : -60.,
        "V_th" : -50.,
        "I_e": 200.,
        "V_reset"   : -60.,
        "V_m"       : -60.
    })

    # ###########################################
    # Population
    # ###########################################
    nodes_ex = Create("iaf_cond_exp", NE)
    nodes_in = Create("iaf_cond_exp", NI)
    nodes = nodes_ex + nodes_in

    # Initialize the membrane potentials
    v = -55.0 + 5.0*numpy.random.normal(size=NE+NI)
    for i, node in enumerate(nodes):
        SetStatus([node], {"V_m": v[i]})

    # ###########################################
    # Projections
    # ###########################################
    # Create the synapses
    w_exc = 6. 
    w_inh = -67.
    SetDefaults("static_synapse", {"delay": 0.1})
    CopyModel("static_synapse", "excitatory",
            {"weight": w_exc})
    CopyModel("static_synapse", "inhibitory", 
            {"weight": w_inh})

    exc_syn_dict={
        "model": "static_synapse",
        "weight": w_exc
    }
    inh_syn_dict={
        "model": "static_synapse",
        "weight": w_inh
    }

    # Create the projections
    conn_dict = {"rule": "one_to_one"}
    A = mmread('ee.wmat')
    Connect(pre=A.row+1, post=A.col+1, syn_spec=exc_syn_dict, conn_spec=conn_dict)

    A = mmread('ei.wmat')
    Connect(A.row+1, A.col+NE+1, syn_spec=exc_syn_dict, conn_spec=conn_dict)

    A = mmread('ie.wmat')
    Connect(A.row+NE+1, A.col+1, syn_spec=inh_syn_dict, conn_spec=conn_dict)

    A = mmread('ii.wmat')
    Connect(A.row+NE+1, A.col+NE+1, syn_spec=inh_syn_dict, conn_spec=conn_dict)

    # Spike detectors
    espikes = Create("spike_detector")
    ispikes = Create("spike_detector")
    Connect(nodes_ex, espikes, 'all_to_all')
    Connect(nodes_in, ispikes, 'all_to_all')

    # ###########################################
    # Simulation
    # ###########################################
    tstart = time.time()
    Simulate(simtime_in_s*1000)
    tstop = time.time() 
    print('Done in', tstop - tstart)


    # ###########################################
    # Data analysis
    # ###########################################
    events_ex   = GetStatus(espikes, "n_events")[0]
    events_in   = GetStatus(ispikes, "n_events")[0]
    print('Total spikes:', events_ex + events_in)
    print('Mean firing rate in the population: ' + str( (events_ex + events_in) / (simtime_in_s*4000.)) + 'Hz')

    return tstop - tstart

if __name__== "__main__":
    "stand alone run"
    nb_threads = 1
    if len(sys.argv) > 1:
        nb_threads = int(sys.argv[1])
    
    configure_and_run_nest(10, nb_threads)
