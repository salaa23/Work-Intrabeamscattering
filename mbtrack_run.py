import numpy as np
from scipy.constants import c, m_e, elementary_charge
import h5py
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
import time
import argparse
import os

print("initializing ...")

def v24(IDs="close", lat="V004", load_lattice=True):
    """
    Fcc-ee booster lattice to load with AT

    Returns
    -------
    ring : Synchrotron object

    """    
    
    h = 1120
    particle = Electron()
    tau = np.array([9.045401802006868, 9.045401802006868, 4.522700901003434])
    sigma_0 = 0.004/c
    sigma_delta = 0.001
    # emit = np.array([6.572332790704266e-07, 1.3144665581408531e-09])    
    # emit = np.array([9.999999999999999e-06/39139.023671477466, 9.999999999999999e-06/39139.023671477466])
    emit = np.array([1.6792275775376196e-11, 1.6792275775376196e-11]) 
    #E0 = 45600000000.0
    # E0 = 20000000000.0 #at injection eV
    tune = np.array([414.225, 410.29])
    chro = np.array([2.057246532, 1.778971585])
    ac = -7.119783162182757e-06
    U0= 1337276.2433950102
    
    if load_lattice:
        if IDs=="close":
            lattice_file = "m4cast/lattice/V24_1FODO_02.mat"
        else:
            lattice_file = "m4cast/lattice/V24_1FODO_02.mat"
    
        # mean values
        # alpha = np.array([0, 0])
        optics = Optics(lattice_file=lattice_file, n_points=1e4)
        
        ring = Synchrotron(h, optics, particle, tau=tau,emit=emit, 
                           sigma_0=sigma_0, sigma_delta=sigma_delta, E0=E0, ac=ac, tune=tune, chro=chro, U0=U0)
    # else:
        # L = 353.97
        # E0 = 2.75e9
        # particle = Electron()
        # ac = 1.0695e-4
        # U0 = 452.6e3
        # tune = np.array([54.2, 18.3])
        # chro = np.array([1.6, 1.6])
        
        # beta = np.array([3.288, 4.003])
        # alpha = np.array([0, 0])
        # dispersion = np.array([0, 0, 0, 0])
        
        # optics = Optics(local_beta=beta, local_alpha=alpha, 
        #               local_dispersion=dispersion)
        # ring = Synchrotron(h, optics, particle, L=L, E0=E0, ac=ac, U0=U0, tau=tau,
        #                emit=emit, tune=tune, sigma_delta=sigma_delta, 
        #                sigma_0=sigma_0, chro=chro)
    
    return ring

def v3588(IDs="close", lat="V004", load_lattice=True):
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
    # [84.4e-12, 84.4e-13]
    
    if load_lattice:
        if IDs=="close":
            lattice_file = "m4cast/lattice/V3588_RING_for_Salah.mat"
        else:
            lattice_file = "m4cast/lattice/V3588_RING_for_Salah.mat"
    
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

def run_mbtrack2(
    n_turns=900,
    n_macroparticles=9000,
    bunch_current=1.2e-2, modelname="PS"
):
    # ring
    ring1 = v24(IDs="open", load_lattice=True)
    # ring2 = v3588(IDs="open", load_lattice=True)
    ring1.f1 = 800000000.0 #RF freq Hz
    ring1.f0 = 3306.823373098939
    ring1.gamma = 39139.023671477466
    ring1.emit[1] = 0.002*ring1.emit[0]
    ring1.tau[0] = ring1.tau[0]/1
    ring1.tau[1] = ring1.tau[1]/1
    ring1.tau[2] = ring1.tau[2]/1
    ring1.L = 90658.71376109403
    ring = ring1

    particle = Electron()
    # bunch
    mybunch = Bunch(
        ring, mp_number=n_macroparticles, current=bunch_current, track_alive=True
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
    V_rf  = 50084569.672473334
    # V_rf = 1.8e6
    # theta = np.arccos(ring.U0 / V_rf)
    theta = -2.458296251434013
    
    rf = RFCavity(ring, m=1, Vc=V_rf, theta=theta)
    # thetas = ((ring.U0/(V_rf * (2 * np.pi * ring.f0)**2)) - np.sqrt(2) / 2) * 1 / 0.5 * 4
    rf2 = RFCavity(ring, m=2, Vc= 0.5*V_rf, theta=np.arccos(np.sqrt(2) / 2))
    ibs = IntrabeamScattering(ring,model=modelname)    

    tracking_elements = [trans_map, long_map, rf, sr, ibs]

    # -----------------------------------------------------------------

    ##ibs track-------------------------------------------------------------------------------------
    job_id = os.environ.get("SLURM_JOB_ID")
    file_path = os.getcwd()
    temps = time.strftime("%y%m%d_%H%M%S", time.localtime())
    file_name = file_path + "/m4cast/outputs/" + modelname + "_" + str(temps) +"_"+ str(job_id)
    monitor = BunchMonitor(1, 1,buffer_size=10, total_size=n_turns, file_name=file_name)
    ###--------------------------------------------------------------------------------------------------------------


    for i in tqdm(range(n_turns)):
        for el in tracking_elements:
            el.track(mybunch)
        monitor.track(mybunch)
    return file_name, temps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script run simulation of V24 lattice using mbtrack2 code, please define the arguments such as:"
                                    )
    parser.add_argument("n_turns", type=int, help="int value for the number of turns")
    parser.add_argument("n_macroparticles", type=int, help="int value for the number of macroparticles")
    parser.add_argument("bunch_current", type=float, help="float value for the bunch current in (A)")
    parser.add_argument("modelname", type=str, help="string argument for the model to use in tracking, choose from: PM, PS, Bane, CIMP")
    


args = parser.parse_args()

print("running sim ...")
file_name, temps = run_mbtrack2(n_turns=args.n_turns,
    n_macroparticles=args.n_macroparticles,
    bunch_current=args.bunch_current, modelname=args.modelname)


print("simulation done using ...")
print(f"number of turns: {args.n_turns}")
print(f"number of macroparticles: {args.n_macroparticles}")
print(f"bunch current(A): {args.bunch_current}")
print(f"model: {args.modelname}")

print("plotting figures ...")
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

job_id = os.environ.get("SLURM_JOB_ID")


with h5py.File(file_name + '.hdf5', 'r') as f:
    # print("Keys: %s" % f.keys())
    group = f["BunchData_1"]
    print(group)
    emit = group["emit"][:]
    current = f["BunchData_1"]["current"][0]
    emit = np.array(emit)


current_dir = os.getcwd()
elements = ["x","y","s"]
for i in range(3):
    plt.figure(figsize=(8,6))
    plt.plot(emit[i,:]*1e12)
    plt.ylabel(f"Emittance epsilon_{elements[i]}(pm)")
    plt.title(f"Emittance of V24 mbt2 at {current*1e3}mA using {args.modelname}")
    plt.xlabel("Number of turns")
    plt.savefig(f"{current_dir}/m4cast/figures/fig_epsi_{elements[i]}_{temps}_{job_id}_cpl.png")

print("Done!")
