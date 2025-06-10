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
src_path = os.path.abspath(os.path.join(os.path.dirname('/Users/ariannameinking/Documents/Brown_Research/quits/docs/ssf_test.ipynb'), '..', 'src'))

if src_path not in sys.path:
    sys.path.insert(0, src_path)


from quits.qldpc_code import *
from quits.circuit import get_qldpc_mem_circuit
from quits.simulation import get_stim_mem_result
from quits.decoder import SSFDecoder
from quits.ldpc_utility import *



n_list = [60, 28,20,12] 
m = 15
dv_list = [3,3,3,3]
dc_list = [12,4,4,4]
dist_list = [4,10,8,6]

code_list = []
log_df = {}

for i, (n, dv, dc, dist) in enumerate(zip(n_list, dv_list, dc_list, dist_list)):
    h = np.loadtxt('/Users/ariannameinking/Documents/Brown_Research/quits/parity_check_matrices/n=%d_dv=%d_dc=%d_dist=%d.txt'%(n, dv, dc, dist), dtype=int)
    curr_code = HgpCode(h, h)
    code_list += [curr_code]
    log_df[f"n={n}, dv={dv}, dc={dc}, dist={dist}"] = []

log_errors_list = []
p_list = np.logspace(-2, -1, 5)
pL_list = []


num_trials_list = np.array([500, 400, 400, 200, 50],dtype=int)

for j,code in enumerate(code_list):
    print(f"Code {j+1}/{len(code_list)}: n={code.hz.shape[1]}, m={code.hz.shape[0]}")
    print(f"classical parameters: n={n_list[j]}, dv={dv_list[j]}, dc={dc_list[j]}, dist={dist_list[j]}")
    print("Starting trials...")
    for i, p in enumerate(p_list):
        for _ in range(num_trials_list[i]):
            print(f"p = {p}, trial {_+1}/{num_trials_list[i]}")
            error = np.random.rand(code.hz.shape[1]) < p
            syndrome = (error@code.hz.T)%2
            SSF_decoder = SSFDecoder(None, code, p, "Z")
            decoded_error = SSF_decoder.decode(syndrome, num_max_iters=200)
            lz_pred = SSF_decoder.logical_error((decoded_error + error)%2)
            log_errors_list += [np.any(lz_pred)]
        pL_list += [sum(log_errors_list)/num_trials_list[i]]
    log_df[f"n={n_list[j]}, dv={dv_list[j]}, dc={dc_list[j]}, dist={dist_list[j]}"] = pL_list

log_df.to_csv('/Users/ariannameinking/Documents/Brown_Research/quits/log_df.csv')





