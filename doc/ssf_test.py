import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.bp_decoder import BpDecoder
from tqdm import tqdm
from scipy.sparse import csc_matrix
import stim
from typing import List, FrozenSet, Dict
import pandas as pd
import warnings
# from decoder import SSFDecoder
import matplotlib.pyplot as plt
import itertools
import sys 
import os

# Get absolute path to ../src (the folder containing qldpc_circuit)
src_path = os.path.abspath(os.path.join(os.path.dirname('/hpc/group/brownlab/am1155/quits/doc/ssf_test.ipynb'), '..', 'src'))

if src_path not in sys.path:
    sys.path.insert(0, src_path)


from quits.qldpc_code import *
from quits.circuit import get_qldpc_mem_circuit
from quits.simulation import get_stim_mem_result
from quits.decoder import SSFDecoder, sliding_window_circuit_mem, detector_error_model_to_matrix
from quits.ldpc_utility import *



n_list = [28,20,12] 
m = 15
dv_list = [3,3,3]
dc_list = [4,4,4]
dist_list = [10,8,6]

code_list = []
log_dict = {}

# for i, (n, dv, dc, dist) in enumerate(zip(n_list, dv_list, dc_list, dist_list)):
#     h = np.loadtxt('../parity_check_matrices/n=%d_dv=%d_dc=%d_dist=%d.txt'%(n, dv, dc, dist), dtype=int)
#     curr_code = HgpCode(h, h)
#     code_list += [curr_code]
#     log_df[f"n={n}, dv={dv}, dc={dc}, dist={dist}"] = []

# first just test with just the 225 code
h = np.loadtxt('../parity_check_matrices/n=%d_dv=%d_dc=%d_dist=%d.txt'%(12, 3, 4, 6), dtype=int)
curr_code = HgpCode(h, h)
code_list += [curr_code]
log_dict[f"n=12, dv=3, dc=4, dist=6"] = []
log_df = pd.DataFrame(log_dict)

log_errors_list = []
p_list = np.logspace(-2, -1, 5)
pL_list = []


num_trials_list = np.array([500, 400, 400, 200, 50],dtype=int)

for j,code in enumerate(code_list):
    pL_list = []
    for i, p in enumerate(p_list):
        num_rounds = 5
        num_trials = num_trials_list[i]
        basis = 'Z'
        W = 1
        F = 1
        circuit = stim.Circuit(get_qldpc_mem_circuit(code, p, p, p, p, num_rounds))
        zcheck_samples, logical_obs_samples = get_stim_mem_result(circuit, num_trials, seed=0)
        dict1 = {'code':code,'p':p, 'error_type':basis}
        dict2 = {'code':code,'p':p, 'error_type':basis}

        DEM = circuit.detector_error_model()
        H, Ls, errors_weights = detector_error_model_to_matrix(DEM)
        logical_pred = sliding_window_circuit_mem(zcheck_samples=zcheck_samples, circuit=circuit, hz=H, lz=Ls, W=W, F=F, decoder1=SSFDecoder, decoder2=SSFDecoder,dict1=dict1, dict2=dict2, error_rate_name1='p', error_rate_name2='p', function_name1='decode', function_name2='decode', tqdm_on=True)

        pL = np.sum((logical_obs_samples - logical_pred).any(axis=1)) / num_trials
        # print('p: %.7f, pL: %.7f'%(p, pL))
        pL_list += [pL]
            
    log_df.loc[i, log_df.columns[j]] = pL_list # used to be j+1, I think this is an artifact from when the first column was the p_list

log_df.to_csv('/Users/ariannameinking/Documents/Brown_Research/quits/doc/log_df_SSF.csv')





