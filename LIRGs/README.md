# Diffuse High-Energy Neutrino Flux from LIRGs


The `diffuse_lirg_prediction.ipynb` notebook presents predictions for the diffuse high-energy neutrino flux from LIRGs. These predictions are based on local estimates derived from a representative sample of GOALS LIRGs. For the theoretical motivation supporting LIRGs as potential high-energy neutrino sources, see our work: Phys. Rev. D 108 (2023) 2, 023015 ([arXiv:2211.09972](https://arxiv.org/abs/2304.01020)).

Source-specific information for the LIRGs is provided in the `Dataframes` folders, while the analysis of the local representative sample is located in the `Completeness` folder. The relevant functions used to compute diffuse flux estimates from these sources are implemented in `diffuse_lirg_extrapolation.py` and `xi_calc.py`. The extrapolation of local predictions to diffuse flux estimates follows the framework outlined in:

- Phys. Rev. D 88** (2013) 12, 121301  
- Phys. Rev. Lett. 116 (2016) 7, 071101


The `diffuse_lirg_fit.py` script provides tools to fit model predictions of diffuse neutrino fluxto the diffuse neutrino flux observed by IceCube, which are stored in thhe `Diffuse_IceCube_data` folder.

