# SWSSB of decohered 1D SPT states

This project is used to calculate SPT states with DMRG, introduce decoherence onto the MPS stat to form MPO mixed states.
Measurements are calculated based on MPS pure state and MPO mixed states.

Measurements include: 

`main_tenpy.py`: String order parameter, Renyi-2 correlator, susceptibility, strange correlator
`compute_one_cmi_gpu.py`: conditional mutual information and negativity. (only for zxz model)

---

# Models
### 1. zxz model with x field 
`tenpy_zxz.py` simulates zxz model with `tenpy`. 

### 2. Haldane spin-1 chain
`tenpy_spin_1.py`

### 3. intrinsic gapless SPT spin chain
`tenpy_zxz_gapless.py`

# Usage
### DMRG simulation:
Run the simulation with:
```
python -u main_tenpy.py --id <id> --h_0 0. --h_step 0.1 --folder set_0 --mode spin_1 --run_mode dmrg
```

### Measure Renyi-2 correlator and other measurements together
```
python -u main_tenpy.py --id "$i" --h_0 "$h_0" --h_step "$h_step" --folder "$folder" --mode "$mode" --run_mode "$run_mode"
```
The simulation is for `h = h_0 + i * h_step`.
DMRG parameters including system length L and bond dimension \chi, quantum channel strength p will be set up in the file `main_tenpy.py`

 ### Measure CMI and negativity
run `compute_one_cmi_gpu.py` directly


