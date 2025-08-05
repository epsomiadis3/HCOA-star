HCOA*: Hierarchical Class-Ordered A* for Navigation in Semantic Environments
=============================================================================

This repository contains the Python implementation of Hierarchical Class-Ordered A* (HCOA*), as presented in the following paper:

HCOA*: Hierarchical Class-Ordered A* for Navigation in Semantic Environments  
Evangelos Psomiadis and Panagiotis Tsiotras  
IEEE Robotics and Automation Letters (RA-L), Accepted August 3, 2025  
https://www.arxiv.org/abs/2505.03128

------------------------------------------------------------------------------

Installation
============

1. Install Required Python Packages
-----------------------------------
From the project root directory, run:

    pip install -r requirements.txt

2. Install Spark-DSG
---------------------
Clone and install the DSG package from the MIT-SPARK/Spark-DSG repository:  
https://github.com/MIT-SPARK/Spark-DSG

Follow their installation instructions for setup.

------------------------------------------------------------------------------

Running the Demos
=================

Step 1: Configure Dataset Path
------------------------------
The **office** and **subway** scenes were created using [Hydra](https://github.com/MIT-SPARK/Hydra) on the [uHumans2 dataset](https://www.mit.edu/~arosinol/datasets/uHumans2/).

To repeat the experiments:
1. Download the uHumans2 dataset and generate the DSG JSON files using Hydra.
2. Adjust the `config.yaml` file and set the `dsg_json_path` to point to the appropriate saved file (either for the office or subway scene).

> ℹ️ If you use the uHumans2 dataset or Hydra, please cite them accordingly.

Step 2: Run Simulation
----------------------
From the project root directory, run:

    python -m OfficeSim
    python -m SubwaySim

Step 3: Create Training Dataset
-------------------------------
This will generate data for training semantic predictors:

    python -m Semantic_Predictor.create_dataset

Outputs are saved in `HCOAStar/Semantic_Predictor/data`.

Step 4: Train Predictive Models
-------------------------------
This trains the GNN and kNN models and compares them to baseline methods:

    python -m Semantic_Predictor.training

Trained models (`bestGNN`, `knn`) are saved in `Semantic_Predictor/`.

------------------------------------------------------------------------------

Path Planning
=============
The proposed HCOA* algorithm and all baseline path planners are located in the `Path_Planner/` directory.

------------------------------------------------------------------------------

Citation
========
If you find this work useful, please consider citing:

@article{Psomiadis2025HCOA,
  author    = {Evangelos Psomiadis and Panagiotis Tsiotras},
  title     = {{HCOA*}: Hierarchical Class-ordered A* for Navigation in Semantic Environments},
  journal   = {IEEE Robotics and Automation Letters (RA-L)},
  year      = {2025},
  month     = {August},
  note      = {Accepted},
}

------------------------------------------------------------------------------

Acknowledgements
================
This work was supported by:
- ARL award DCIST CRA W911NF-17-2-0181
- ONR award N00014-23-1-2304

