# Graph-fused Lasso for Clustering in Networks with Time-varying Nodal Attributes

Yik Lun Kei, Oscar Hernan Madrid Padilla, Rebecca Killick, James D. Wilson, Xi Chen, Robert Lund

## Overview
This repository contains code for the **Graph-fused Lasso** (GFL) method for clustering in networks with time-varying nodal attributes.

Use CP_GFL_demo.ipynb to reproduce the results in paper. 

## Folder Structure

- `CD_GFL_sim.py`: Python script to run the algorithm on simulated data.
- `CD_GFL_real.py`: Python script to run the algorithm on real-world data.
- `data/`: Contains the simulated data, real data, and R scripts for figure generation.
- `result/`: Output folder for the results of the algorithm.
- `competitor/`: Contains the scripts and results from the competitor methods.

## How to Reproduce the Results

1. **Simulated Data**  
   To reproduce the results in the paper, use the Jupyter notebook `CPD_demo.ipynb`. For example, use `sim_data_s1.py` to generate simulated data, which will be saved to the `data/` folder .

2. **Nodal Clustering**  
   To apply the GFL method for nodal clustering, use the script `CD_GFL_sim.py`. The output will be saved at the `result/` folder. 

3. **Real Data Experiments**  
   For read data experiments, run the script `CD_GFL_real.py`. The result will be saved at the `result/` folder. Additionally, the R scripts in the folder of the two data set can be used to reproduce the figures shown in the paper.
