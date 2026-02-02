# Decoder-only Clustering in Graphs with Dynamic Attributes

Yik Lun Kei, Oscar Hernan Madrid Padilla, Rebecca Killick, James D. Wilson, Xi Chen, Robert B. Lund\
Under Review

## Overview
This repository contains the code of the **graph-fused Lasso** (GFL) method for clustering in networks with time-varying nodal attributes.

The framework includes **prior distributions** for low-dimensional representations and a **decoder** that bridges the latent representations and observed sequences. The graph-fused Lasso regularization is imposed on the prior parameters, and the resulting optimization problem is solved via the **alternating direction method of multipliers** (ADMM).

## Folder Structure

- `CD_GFL_sim.py`: Python script to run the algorithm on simulated data.
- `CD_GFL_real.py`: Python script to run the algorithm on real-world data.
- `data/`: Contains the simulated data, real data, and R scripts for figure generation.
- `result/`: Output folder for the results of the algorithm.
- `competitor/`: Contains the scripts and results from the competitor methods.

## How to Reproduce the Results

To reproduce the results in the paper, use the Jupyter notebook `CD_GFL_demo.ipynb`.

1. **Simulated Data**  
    The data for the simulation study is already generated and existed in the `data/` folder. If needed, for example, use `sim_data_s1.py` to generate the simulated data for Scenario 1, which will be saved to the `data/` folder. 

2. **Simulation Study**  
   To apply the GFL method for nodal clustering, use the script `CD_GFL_sim.py`. The script uses the data saved in the `data/` folder, and the output will be saved at the `result/` folder. 

3. **Real Data Experiments**  
   For read data experiments, run the script `CD_GFL_real.py`. The result will be saved at the `result/` folder. Additionally, the R scripts in the folder of the two data set can be used to reproduce the figures shown in the paper.
