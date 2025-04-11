import numpy as np
from scipy.constants import c, m_e, elementary_charge
import h5py as hp
import matplotlib.pyplot as plt
from mbtrack2 import Synchrotron, Electron
from mbtrack2.utilities import Optics
from mbtrack2.impedance.wakefield import WakeField
from mbtrack2.tracking import LongitudinalMap, SynchrotronRadiation, TransverseMap
from mbtrack2.tracking import IntrabeamScattering
from mbtrack2.tracking import Beam, Bunch, WakePotential
from mbtrack2.tracking import RFCavity, SynchrotronRadiation
from mbtrack2.tracking.monitors import BunchMonitor, WakePotentialMonitor
from mbtrack2.tracking.feedback import FIRDamper, ExponentialDamper
import at
from tqdm import tqdm
from time import time
import scipy.integrate as quad

def v2366(IDs="close", lat="V004", load_lattice=True):
    """
    TDR lattice using V2366_V004_Physical_Aperture.m

    Returns
    -------
    ring : Synchrotron object

    """    
    
    h = 416
    particle = Electron()
    tau = np.array([7.68e-3, 14.14e-3, 12.18e-3])
    sigma_0 = 9e-12
    sigma_delta = 9.07649e-4
    emit = np.array([84.4e-12, 84.4e-13])
    
    if load_lattice:
        if IDs=="close":
            lattice_file = "V3588_RING_for_Salah.mat"
        else:
            lattice_file = "V3588_RING_for_Salah.mat"
    
        # mean values
        alpha = np.array([0, 0])
        optics = Optics(lattice_file=lattice_file, local_alpha=alpha, n_points=1e4)
        
        ring = Synchrotron(h, optics, particle, tau=tau, emit=emit, 
                           sigma_0=sigma_0, sigma_delta=sigma_delta)
    else:
        L = 353.97
        E0 = 2.75e9
        particle = Electron()
        ac = 1.0695e-4
        U0 = 452.6e3
        tune = np.array([54.2, 18.3])
        chro = np.array([1.6, 1.6])
        
        beta = np.array([3.288, 4.003])
        alpha = np.array([0, 0])
        dispersion = np.array([0, 0, 0, 0])
        
        optics = Optics(local_beta=beta, local_alpha=alpha, 
                      local_dispersion=dispersion)
        ring = Synchrotron(h, optics, particle, L=L, E0=E0, ac=ac, U0=U0, tau=tau,
                       emit=emit, tune=tune, sigma_delta=sigma_delta, 
                       sigma_0=sigma_0, chro=chro)
    
    return ring
    
def model_ring():
    h = 416 # Harmonic number of the accelerator.
    L = 353.97 # Ring circumference in [m].
    E0 = 2.75e9 # Nominal (total) energy of the ring in [eV].
    particle = Electron() # Particle considered.
    ac = 1.0695e-4
    U0 = 452.6e3# Energy loss per turn in [eV].
    tau = np.array([7.68e-3, 14.14e-3, 12.18e-3]) #horizontal, vertical and longitudinal damping times in [s].
    tune = np.array([54.2, 18.3])
    emit = np.array([84.4e-12, 84.4e-13])
    sigma_0 = 9e-12
    sigma_delta = 9.07649e-4
    chro = np.array([1.6, 1.6])
    beta = np.array([3.288, 4.003])
    alpha = np.array([0, 0])
    dispersion = np.array([0, 0, 0, 0])
    optics = Optics(local_beta=beta, local_alpha=alpha, 
                      local_dispersion=dispersion)
    ring3 = Synchrotron(h=h, optics=optics, particle=particle, L=L, E0=E0, ac=ac,
                       U0=U0, tau=tau, emit=emit, tune=tune,
                       sigma_delta=sigma_delta, sigma_0=sigma_0, chro=chro)
    return ring3
#Particles number
def run_mbtrack2(
    n_turns=900,
    n_macroparticles=9000,
    bunch_current=1.2e-2, modelname="PS"
):
    # ring
    ring3 = model_ring()
    ring = v2366(IDs="open")
    ring2 = ring
    ring2.emit[1] = .3*ring2.emit[0]
    ring2.tau[0] = ring2.tau[0]/1
    ring2.tau[1] = ring2.tau[1]/1
    ring2.tau[2] = ring2.tau[2]/1

    particle = Electron()
    # bunch
    mybunch = Bunch(
        ring2, mp_number=n_macroparticles, current=bunch_current, track_alive=True
    )
    np.random.seed(42)
    mybunch.init_gaussian()
    # offset 
    # mybunch['x'] = +1e-3
    # mybunch['y'] = +1e-3
    #Tracking elements
    long_map = LongitudinalMap(ring)
    sr = SynchrotronRadiation(ring, switch=[1, 1, 1])
    trans_map = TransverseMap(ring)
    V_rf  = 1.8e6 #1.8e6
    rf = RFCavity(ring2, m=1, Vc=V_rf, theta=np.arccos(ring.U0 / V_rf))
    # thetas = ((ring.U0/(V_rf * (2 * np.pi * ring.f0)**2)) - np.sqrt(2) / 2) * 1 / 0.5 * 4
    rf2 = RFCavity(ring2, m=2, Vc= 0.5*V_rf, theta=np.arccos(np.sqrt(2) / 2))
    ibs = IntrabeamScattering(ring, mybunch)
    ibs.current_model = modelname
    

    tracking_elements = [trans_map, long_map, rf, sr, ibs]

    ##ibs track-------------------------------------------------------------------------------------
    monitor = BunchMonitor(1, 10,buffer_size=100, total_size=n_turns, file_name=modelname)
    ###--------------------------------------------------------------------------------------------------------------

    for i in tqdm(range(n_turns)):
        for el in tracking_elements:
            el.track(mybunch)
        monitor.track(mybunch)


run_mbtrack2(n_turns=500,
    n_macroparticles=1e6,
    bunch_current=1.2e-3, modelname="Bane")
