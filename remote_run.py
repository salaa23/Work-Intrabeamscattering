from utilities_remote import _collective_
import os

os.environ["CFLAGS"] = "-w -Wno-maybe-uninitialized"

_inputs = {
    'mode' : 'z',
    'n_macroparticles' : int(10_000), # number of macroparticles
    'n_turns' : int(30_000), # number of turns
    #'optics_file' : fodo_json, # optics file in json format
    'ibs' : True, # boolean
    'wake' : True, # boolean
    'detuning' : True, # boolean
    'dispersion' : True, # boolean
    'gaussian_noise' : True, # boolean
    'origin' : 'map', # can be "map" or "lattice" for tracking through the whole lattice or the linear map
    'kernel': 'cpu', # kernel to be used, can be 'cpu' or 'gpu' or 'omp'
    'comp' : 'bench', # can be 'bench' or 'ccin2p3' or 'lxplus' or 'criann' or 'local'
    'dir_name' : 'delete_me',
    'comment': 'testing the classes',
    'bunch': 'matched', # can be 'matched' or 'unmatched'
    }

func = _collective_(_inputs=_inputs)
