'''
Author: Adnan Ghribi, Salah Feddaoui
Date: 2023-10-03
Description: This file contains utility functions for the v12 classes.
Date de dernière modification: 2025-04-10
'''

# Import the necessary packages
import json, sys, os, warnings, random, string, shutil
import numpy as np
from astropy.io import ascii
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from tqdm import tqdm
import scipy.constants as cons
import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo
import xcoll as xc
import xwakes as xw
import time
#

def dump_data(filename, output):
    with open(filename, 'w') as json_file:
        json.dump(output, json_file)


def make_unique(filename):
    make_unique = lambda filename: f"{filename.rsplit('.', 1)[0]}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}.{filename.rsplit('.', 1)[1]}"
    new_name = make_unique(filename)
    return new_name


class beam_param:
    """
    Class to define the beam parameters from input file
    """
    def __init__(self, mode, **kw):
        if mode == None:
            print('No mode entered -> default = z')
            mode = 'z'
        self.filename()
        self.read_param(mode)
        self.set_param()

    def filename(self):
        # parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
        input_dir = '_inputs/'#parent_dir + '/input_data/'
        self.filename = input_dir + 'Booster_parameter_table.json'
        
    def read_param(self, mode):
        inputs_tab = json.load(open(self.filename))
        print(inputs_tab.keys())

        # define variables
        self.C = inputs_tab['C']['value'] # circumference [m]
        self.Np = inputs_tab['Np'][mode] # number of particles per bunch
        self.Nb = inputs_tab['Nb'][mode] # number of bunches
        self.Etot = inputs_tab['E']['injection'] # energy at injection [eV]
        self.epsnx = inputs_tab['bunch']['epsnx']['value'] # normalised horizontal emittance [m]
        self.epsny = inputs_tab['bunch']['epsny']['value'] # normalised horizontal emittance [m]
        self.sigmaz = inputs_tab['bunch']['sigmaz']['value'] # bunch length at injection [m]
        self.sigmae = inputs_tab['bunch']['sigmae']['value'] # energy spread at injection
        self.Qx = inputs_tab['optics']['Qx'][mode] # horizontal tune
        self.Qy = inputs_tab['optics']['Qy'][mode] # vertical tune
        self.chix = inputs_tab['optics']['chix'][mode] # horizontal chromaticity
        self.chiy = inputs_tab['optics']['chiy'][mode] # horizontal chromaticity
        self.alpha = inputs_tab['optics']['alpha'][mode] # momentum compaction
        self.I2 = inputs_tab['optics']['I2'][mode] # 2nd synchrotron integral
        self.I3 = inputs_tab['optics']['I3'][mode] # 3rd synchrotron integral
        self.I5 = inputs_tab['optics']['I5'][mode] # 5th synchrotron integral
        self.I6 = 0 # 6th synchrotron integral
        self.dpt = inputs_tab['optics']['dpt'][mode] # maximum energy acceptance at injection
        self.damp_xy = inputs_tab['optics']['damp_xy'][mode] # transverse damping time at injection energy
        self.damp_s = inputs_tab['optics']['damp_s'][mode] # longitudinal damping time at injection energy
        self.coupling = inputs_tab['optics']['coupling'][mode] # horizontal vertical coupling
        self.Cq = 3.8319e-13
        self.Cgamma = 8.846e-05
        self.Erest = 510998.9499961642 # rest energy [eV]
        self.Egain = 0 # energy gain [eV]
        self.freq = inputs_tab['RF']['RF_freq'][mode] # RF frequency [Hz]
        self.Vtot = inputs_tab['RF']['Vtot'][mode] # total cavities voltage [eV]
        self.Qs = inputs_tab['RF']['Qs_inj'][mode] # synchronous tune at injection
        self.phi_s = inputs_tab['RF']['phis_inj'][mode] # synchronous phase at injection
        
    def set_param(self):
        self.lambdaRF = cons.c/self.freq # RF wavelength
        self.frev = cons.c/self.C # revolution frequency
        self.h = self.freq/self.frev # Schottky number
        self.U0 = self.Cgamma*(self.Etot*1e-9)**4/2/np.pi*self.I2*1e9 # Synchrotron energy loss per turn
        self.gamma = self.Etot / self.Erest
        self.sig_e_eq = np.sqrt(self.Cq*self.gamma**2*self.I3/(2*self.I2))#*(self.Etot*1e-9)**2)
        self.emit_eq = self.Cq * self.gamma**2 * self.I5 / self.I2 # geometrical equilibrium emittance
        self.epsnx_eq = self.emit_eq * self.gamma # normalized equilibrium emittance X
        self.epsny_eq = self.epsnx_eq * 2e-3 # normalized equilibrium emittance Y
        self.eta = 1/self.gamma**2-self.alpha # momentum compaction

class _collective_():
    '''
    Class for tracking with collective effects
    Usage:
    _inputs = {
    'bunch_intensity' : bunch_intensity,
    'n_turns' : n_turns, # number of turns
    'n_macroparticles' : n_macroparticles,
    'optics_file' : fodo_json, # optics file in json format
    'ibs' : ibs, # boolean
    'wake' : wake, # boolean
    'detuning' : detuning, # boolean
    'dispersion' : dispersion, # boolean
    'origin' : origin, # can be "map" or "lattice" for tracking through the whole lattice or the linear map
    'kernel': kernel, # kernel to be used, can be 'cpu' or 'gpu' or 'omp'
    'comp' : 'bench',
    'comment': 'no wake, no gaussian noise, no dispersion, no detuning'
}
    '''
    # initialize the class
    def __init__(self, _inputs=None, **kw):
        print ('== loading inputs ==')
        self._set_defaults()
        if _inputs is not None:
            for key, value in _inputs.items():
                setattr(self, key, value)
                print(key, value)
        else:
            print('Some input parameters are missing')
        print('\n== setting defaults ==')
        print('\n== setting parameters ==')
        self._inputs = _inputs
        self._set_params()
        self._set_path()
        print('\n== setting line ==\n')
        self._set_line_param()
        print('\n== calculating one turn map ==\n')
        self._set_map()
        print('\n== setting IBS ==\n')
        self._set_ibs()
        print('\n== creating bunch ==\n')
        self._set_bunch()
        # print('\n== setting wake ==\n')
        # self._set_wake()
        print('\n== setting monitor ==\n')
        self._set_monitor()
        print('\n== setting radiation==\n')
        self._set_radiation()
        print('\n== Tracking ... ==\n')
        self._track()

        
    def _set_defaults(self):
        # input files
        self.input_dir = '_inputs/'
        self.optics_file = self.input_dir + 'heb_ring_withcav.json' # optics file
        self.mass =xp.ELECTRON_MASS_EV # mass of the particle
        self.q0 = -1 # charge of the particle
        self.wakefile = self.input_dir + 'heb_wake_round_cu_30.0mm.csv' # wake file
        # slicing parameters
        self.n_slices_wake = 500
        self.slicing_mode = 'fixed_cuts' # can be 'fixed_cuts' or 'from_first_to_last_particles'
        self.fixed_cuts_perc_min_max = 0.50
        # chunk size for tracking
        self.chunk_size = 1
        # IBS parameters
        self.n_slices_ibs = 50
        self.ibs_formalism = 'nagaitsev' # can be 'nagaitsev' or 'kinetic'
        # bunch parameters
        self.bunch = 'matched' # can be 'matched' or 'unmatched'
        
    def _set_params(self, **kw):
        self.param = beam_param(self.mode)
        # ICI ON APPELE BEAM PARAM 
        self.param.set_param()
        self.param.read_param(self.mode)
        print(vars(self.param))
        ###########
        for key, value in self.param.__dict__.items():
            setattr(self, key, value) 
        self.limit_z = 3 * self.sigmaz
        if self.kernel == 'gpu':
            self.context = xo.ContextCupy()
        elif self.kernel == 'omp':
            self.context = xo.ContextCpu(omp_num_threads="auto")
        else:
            self.context = xo.ContextCpu()
        self.n_chunk = int(np.ceil(self.n_turns/self.chunk_size))
        self.lag_rf = 180
    
    def _set_path(self):
        parent_dir = os.getcwd()#os.path.dirname(os.getcwd())
        print(parent_dir)
        self.input_dir = parent_dir + '/_inputs'
        if self.dir_name is not None:
            self.dir_ = self.dir_name + '/'
        else:
            self.dir_ = 'part' + str(int(self.Np/1e10)) + 'e10_freq' + str(int(self.freq*1e-6)) + '_sz'+str(self.sigmaz) + '_mode_'+self.mode+'/'
        if self.comp == 'feynman':
            self.res_directory = parent_dir + '/.results/results_feynman/' + self.dir_
        if self.comp == 'ccin2p3':
            self.res_directory = parent_dir + '/.results/results_ccin2p3/' + self.dir_
        elif self.comp == 'lxplus':
            self.res_directory = '/eos/project-f/fcc-ee-ce/Salah/_dev_collective_effects/fccee_collective_effects/pyhdt_heb/heb_single_bunch/.results/results_lxplus/' + self.dir_
        elif self.comp == 'bench':
            self.res_directory = parent_dir + '/_outputs/'+ self.dir_
        self.fig_directory = self.res_directory + 'figures/'
        self.data_directory = self.res_directory +'data/'
        if os.path.exists(self.res_directory)==False:
            os.makedirs(self.res_directory)
        if os.path.exists(self.fig_directory)==False:
            os.makedirs(self.fig_directory)
        if os.path.exists(self.data_directory)==False:
            os.makedirs(self.data_directory)
        if os.path.exists(self.data_directory + 'moments')==False:
            os.makedirs(self.data_directory + 'moments')
        delete_folder = False
        if os.path.exists(self.fig_directory) and delete_folder:    
            shutil.rmtree(self.fig_directory)
            os.makedirs(self.fig_directory)
        folders_names = ['', 'phasespaces_xxp', 'phasespaces_yyp', 'phasespaces_zdp',
                     'profile_x', 'profile_y', 'profile_z'
                     ]
        for i in folders_names:
            try:
                os.makedirs(self.fig_directory+i)
            except:
                pass 
        
    def _set_line_param(self):
        self.line = xt.Line.from_json(self.optics_file) # import lattice
        self.particle_ref = xp.Particles(mass0=self.mass, q0=self.q0, gamma0=self.gamma) # define reference particle
        self.line.particle_ref  = self.particle_ref # assign reference particle to the line
        self.line.slice_thick_elements( slicing_strategies=[
        # Slicing with thin elements
            xt.Strategy(slicing=xt.Teapot(2)), # (1) Default applied to all elements
            xt.Strategy(slicing=xt.Teapot(3), element_type=xt.Bend), # (2) Selection by element type
            xt.Strategy(slicing=xt.Teapot(5), element_type=xt.Quadrupole),  # (4) Selection by element type
            xt.Strategy(slicing=xt.Teapot(3), element_type=xt.Sextupole),  # (4) Selection by element type
        ])
        print('building tracker')
        self.line.build_tracker() # build the tracker
        print('configuring radiation')
        self.line.configure_radiation(model='mean') # configure radiation
        self.env = self.line.env
        print('calculating twiss')
        self.tw = self.line.twiss(method="6d", eneloss_and_damping=True) # get twiss parameters
        self.C = self.tw.s[-1] # circumference
        self.qx = self.tw.qx # horizontal tune
        self.qy = self.tw.qy # vertical tune
        self.dqx = self.tw.dqx # horizontal chromaticity
        self.dqy = self.tw.dqy # vertical chromaticity
        self.eneloss_turn = self.tw.eneloss_turn
        df = self.tw.to_pandas() # convert to pandas dataframe
        betax_0 = self.C / (2*np.pi*self.tw.qx) # betax from the tune
        # betay_0 = self.param.C / (2*np.pi*self.params.Qy) # betax from the tune
        # the 10 closest values to the betax_0
        df['distance'] = abs(df.betx - betax_0)
        closest = df.nsmallest(10, 'distance')
        closest = closest.drop('distance', axis=1)
        self.index = closest.dx.idxmin()
        self.betax = closest.betx[self.index] # horizontal beta function from the twiss
        self.betay = closest.bety[self.index] # vertical beta function from the twiss
        self.alphax = closest.alfx[self.index] # horizontal alpha function from the twiss
        self.alphay = closest.alfy[self.index] # vertical alpha function from the twiss
        if self.dispersion==True:
            self.dx = closest.dx[self.index] # horizontal dispersion from the twiss
            self.dy = closest.dy[self.index] # vertical dispersion from the twiss
        else:
            self.dx = 0
            self.dy = 0
        # get amplitude detuning coefficients
        if self.detuning==True:
            print('Extracting detuning coefficients')
            det_= self.line.get_amplitude_detuning_coefficients(
                nemitt_x=self.epsnx, 
                nemitt_y=self.epsny, 
                num_turns=500, 
                a0_sigmas=0.01, 
                a1_sigmas=0.1, 
                a2_sigmas=0.2
            )
            self.det_xx = det_['det_xx']
            self.det_yy = det_['det_yy']
            self.det_xy = det_['det_xy']
            self.det_yx = det_['det_yx']
        else:
            self.det_xx = 0
            self.det_yy = 0
            self.det_xy = 0
            self.det_yx = 0
        # get synchrotron parameters
        self.damping_rate_emit_h = 2 * self.tw.damping_constants_turns[0] # horizontal damping rate
        self.damping_rate_emit_v = 2 * self.tw.damping_constants_turns[1] # horizontal damping rate
        self.damping_rate_emit_zeta = 2 * self.tw.damping_constants_turns[2] # longitudinal damping rate
        # Compute gaussian noise amplitudes to model quantum excitation
        if self.gaussian_noise==True:
            self.gauss_noise_ampl_px = 2 * np.sqrt(self.tw.eq_gemitt_x * self.damping_rate_emit_h / self.tw.betx[self.index])
            self.gauss_noise_ampl_x = 0
            self.gauss_noise_ampl_py = 2 * np.sqrt(self.tw.eq_gemitt_y * self.damping_rate_emit_h / self.tw.bety[self.index])
            self.gauss_noise_ampl_y = 0.
            self.gauss_noise_ampl_delta = 2 * np.sqrt(self.tw.eq_gemitt_zeta * self.damping_rate_emit_zeta / self.tw.bets0)
    
    def _set_map(self):
        self.map =  xt.LineSegmentMap(
                    length=self.C,
                    qx=self.qx,
                    qy=self.qy,
                    dqx=self.dqx,
                    dqy=self.dqy,
                    momentum_compaction_factor=self.alpha,
                    betx=self.betax,
                    bety=self.betay,
                    alfx=self.alphax,
                    alfy=self.alphay,
                    dx = self.dx,
                    dy = self.dy,
                    det_xx=self.det_xx,
                    det_xy=self.det_xy,
                    det_yx=self.det_yx,
                    det_yy=self.det_yy,  
                    damping_rate_x=  self.damping_rate_emit_h,
                    damping_rate_y=  self.damping_rate_emit_v,
                    # In longitudinal all damping goes on the momentum
                    damping_rate_zeta=0,
                    damping_rate_pzeta=2 * self.tw.damping_constants_turns[2],
                    gauss_noise_ampl_px=self.gauss_noise_ampl_px,
                    gauss_noise_ampl_py=self.gauss_noise_ampl_py,
                    gauss_noise_ampl_pzeta=self.gauss_noise_ampl_delta,
                    energy_increment           = -1 * self.eneloss_turn,#M.U0,
                    longitudinal_mode          = 'nonlinear', # needs to be commented for 4D tracking + uncomment betas/qs
                    voltage_rf                 = [self.Vtot], # needs to be commented for 4D tracking + uncomment betas/qs
                    frequency_rf               = [self.freq], # needs to be commented for 4D tracking + uncomment betas/qs
                    lag_rf                     = [180 - np.rad2deg(np.arcsin(self.eneloss_turn/self.Vtot))], # needs to be commented for 4D tracking + uncomment betas/qs
                )
        
        ring_map_no_excit = self.map.copy()
        self.env.elements['ring_map'] = self.map
        ring_map_no_excit.gauss_noise_matrix = 0
        self.lring = xt.Line(elements=[ring_map_no_excit])
        self.lring._needs_rng = True
        self.lring.particle_ref = self.particle_ref.copy()
        tw_check = self.lring.twiss()
        self.lring.correct_trajectory(twiss_table=self.lring.twiss4d())
        
    def _set_ibs(self):
        if self.ibs == True:
            ibs_kick = xf.IBSKineticKick(num_slices=self.n_slices_ibs)
            self.line.configure_intrabeam_scattering(element=ibs_kick, 
                                                     name="ibskick", 
                                                     index=self.index, 
                                                     update_every=1)
            if self.ibs_formalism == 'nagaitsev':
                ibs_kick = xf.IBSAnalyticalKick(formalism="B&M", num_slices=self.n_slices_ibs)
            elif self.ibs_formalism == 'kinetic':
                ibs_kick = xf.IBSKineticKick(num_slices=self.n_slices_ibs)
            self.line_map = self.env.new_line(name='line_map', components=['ring_map', 'ibskick'])
        else:
            self.line_map = self.env.new_line(name='line_map', components=['ring_map'])
        
        self.line_map.particle_ref = self.particle_ref.copy()
        self.line_map._needs_rng = True
    
    def _set_bunch(self):
        rng = np.random.RandomState(42)
        x_norm = rng.randn(self.n_macroparticles)
        px_norm = rng.randn(self.n_macroparticles)
        y_norm = rng.randn(self.n_macroparticles)
        py_norm = rng.randn(self.n_macroparticles)
        zeta = self.sigmaz * (rng.randn(self.n_macroparticles))
        delta = self.sigmae * (rng.randn(self.n_macroparticles))
        if self.bunch == 'matched':
            self.particles = xp.generate_matched_gaussian_bunch(
                num_particles=self.n_macroparticles,
                nemitt_x=self.epsnx,
                nemitt_y=self.epsny,
                sigma_z=self.sigmaz,
                total_intensity_particles=self.Np,
                line=self.lring,
                _context=self.context,
            )
        else:
            self.particles = self.lring.build_particles(
                _context=xo.ContextCpu(), 
                _buffer=None, 
                _offset=None,
                particle_ref=self.particle_ref,
                zeta=zeta, 
                delta=delta,
                x_norm=x_norm, 
                px_norm=px_norm,
                y_norm=y_norm, 
                py_norm=py_norm,
                nemitt_x=self.epsnx, 
                nemitt_y=self.epsny,
                weight=self.Np/self.n_macroparticles)
        
    def _set_wake(self):
        # slicing strategy
        if self.slicing_mode == 'from_first_to_last_particles':
            initial_cut_tail_z = np.min(self.particles.zeta) 
            initial_cut_head_z = np.max(self.particles.zeta)
        elif self.slicing_mode == 'fixed_cuts':
            initial_cut_tail_z = np.min(self.particles.zeta) - 0.5*(np.max(self.particles.zeta)-np.min(self.particles.zeta))
            initial_cut_head_z = np.max(self.particles.zeta) + 0.5*(np.max(self.particles.zeta)-np.min(self.particles.zeta))
        #
        T = ascii.read(self.wakefile)
        temp_file = make_unique('temp.txt')
        np.savetxt(temp_file, np.transpose(np.array(T)))
        wake_df = xw.read_headtail_file(
            temp_file, ["time", "longitudinal", "dipole_x", "dipole_y"])
        wf_xw = xw.WakeFromTable(wake_df,
            ["longitudinal", "dipole_x", "dipole_y"])
        wf_xw.configure_for_tracking(
            zeta_range=(initial_cut_tail_z, initial_cut_head_z),
            num_slices=self.n_slices_wake)
        os.remove(temp_file)
        self.line_map.append('wake_field', wf_xw)
    
    def _set_monitor(self):
        self.emit_mon = xc.EmittanceMonitor.install(line=self.line_map, 
                                name="EmittanceMonitor", 
                                at=0, stop_at_turn=self.n_turns)
    
    def _set_radiation(self):
        self.line_map.build_tracker()
        self.line_map.configure_radiation(model='quantum')
        
    def _track(self):
        pbar = tqdm(range(self.n_chunk))
        mean_x  = []
        mean_y  = []
        mean_z  = []
        mean_e = []
        sigma_x = []
        sigma_y = []
        sigma_z = []
        sigma_e = []
        for i_chunk in pbar:
            pbar.set_description(f'Chunk {i_chunk+1}/{self.n_chunk}')
            monitor = xt.ParticlesMonitor(_context= self.context,
                    start_at_turn=i_chunk*self.chunk_size, 
                    stop_at_turn=(i_chunk+1)*self.chunk_size,
                    num_particles=self.n_macroparticles)
            self.line_map.track(self.particles, 
                                num_turns=self.chunk_size, 
                                turn_by_turn_monitor=monitor, 
                                with_progress=False)
            mean_x[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]   = np.average(monitor.x,axis=0)
            mean_y[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]   = np.average(monitor.y,axis=0)
            mean_z[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]   = np.average(monitor.zeta,axis=0)
            mean_e[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]   = np.average(monitor.delta,axis=0)
            sigma_x[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(monitor.x,axis=0)
            sigma_y[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(monitor.y,axis=0)
            sigma_z[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(monitor.zeta,axis=0)
            sigma_e[i_chunk*self.chunk_size:(i_chunk+1)*self.chunk_size]  = np.std(monitor.delta,axis=0)
            
        output = {
            'epsx' : {
                'value' : self.emit_mon.gemitt_x.tolist(),
                'label' : r'$\epsilon_x$'
            },
            'epsy' : {
                'value': self.emit_mon.gemitt_y.tolist(),
                'label': r'$\epsilon_y$'
            },
            'meanx' : {
                'value': list(mean_x),
                'label' : r'$\bar{x}$'
            },
            'meany' : {
                'value': list(mean_y),
                'label' : r'$\bar{y}$'
            },
            'meanz' : {
                'value': list(mean_z),
                'label' : r'$\bar{z}$'
            },
            'meane' : {
                'value': list(mean_e),
                'label' : r'$\bar{e}$'
            },
            'sigmax' : {
                'value': list(sigma_x),
                'label' : r'$\sigma_x$'
            },
            'sigmay' : {
                'value': list(sigma_y),
                'label' : r'$\sigma_y$'
            },
            'sigmaz' : {
                'value': list(sigma_z),
                'label' : r'$\sigma_z$'
            },
            'sigmae' : {
                'value' : list(sigma_e),
                'label' : r'$\sigma_e$'
            },
            'inputs' : self._inputs,
            'parameters': vars(self.param),
            # 'comment' : self.comments
        }
        # print(filename)
        temps = time.strftime("%y%m%d_%H%M%S", time.localtime())
        print(type(self.param))
        print("Creating file...")
        dump_data('/home/salahfd/_outputs/output_'+temps+'.json', output)
        # dump_data(self.res_directory + self.Ofilename, output)
        print("Done!")
