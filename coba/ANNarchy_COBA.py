from ANNarchy import *
from scipy.io import mmread

def configure_and_run_ann(simtime_in_s, num_threads):
    """
    Build up the network and run for a period *simtime_in_s*.
    The function returns the simulation time for 1 run without
    compilation time including time for record.
    """
    # ###########################################
    # Configuration
    # ###########################################
    #clear()
    setup(dt=0.1, seed=98765, num_threads=num_threads)

    # ###########################################
    # Network parameters
    # ###########################################
    NE = 3200        # Number of excitatory cells
    NI = 800         # Number of inhibitory cells
    duration = 10.0  # Total time of the simulation in seconds

    # ###########################################
    # Neuron model
    # ###########################################
    COBA = Neuron(
        parameters="""
            El = -60.0  : population
            Vr = -60.0  : population
            Erev_exc = 0.0  : population
            Erev_inh = -80.0  : population
            Vt = -50.0   : population
            tau = 20.0   : population
            tau_exc = 5.0   : population
            tau_inh = 10.0  : population
            I = 20.0 : population
        """,
        equations="""
            tau * dv/dt = (El - v) + g_exc * (Erev_exc - v) + g_inh * (Erev_inh - v ) + I

            tau_exc * dg_exc/dt = - g_exc 
            tau_inh * dg_inh/dt = - g_inh 
        """,
        spike = """
            v > Vt
        """,
        reset = """
            v = Vr
        """,
        refractory = 5.0
    )

    # ###########################################
    # Population
    # ###########################################
    P = Population(geometry=NE+NI, neuron=COBA)
    Pe = P[:NE]
    Pi = P[NE:]
    P.v = Normal(-55.0, 5.0)

    # ###########################################
    # Projections
    # ###########################################
    A = mmread('ee.wmat')
    Cee = Projection(pre=Pe, post=Pe, target='exc')
    Cee.connect_from_sparse(A.tocsr())

    A = mmread('ei.wmat')
    Cei = Projection(pre=Pe, post=Pi, target='exc')
    Cei.connect_from_sparse(A.tocsr())

    A = mmread('ii.wmat')
    Cii = Projection(pre=Pi, post=Pi, target='inh')
    Cii.connect_from_sparse(A.tocsr()) 

    A = mmread('ie.wmat')
    Cie = Projection(pre=Pi, post=Pe, target='inh')
    Cie.connect_from_sparse(A.tocsr())

    import os
    os.system("rm -rf annarchy")
    compile()

    # ###########################################
    # Simulation
    # ###########################################
    m = Monitor(P, 'spike')
    tstart = time.time()
    simulate(duration* 1000.0)
    tstop = time.time()
    print('Done in', tstop - tstart)

    # ###########################################
    # Data analysis
    # ###########################################
    t, n = m.raster_plot()
    print('Number of spikes:', len(t))
    print('Mean firing rate in the population: ' + str(len(t) / (duration*4000.)) + 'Hz')

    # from pylab import *
    # plot(t, n, '.', markersize=2)
    # xlabel('Time (ms)')
    # ylabel('Neuron index')
    # show()

    return tstop - tstart

if __name__== "__main__":
    "stand alone run"
    nb_threads = 1
    if len(sys.argv) > 1:
        nb_threads = int(sys.argv[1])
    
    configure_and_run_ann(10, nb_threads)
