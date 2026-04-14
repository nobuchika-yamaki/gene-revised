This repository contains the code used to reproduce all numerical results presented in the study on non-reducibility of symmetry-breaking attractors in a minimal regulatory model.

The implementation evaluates both a symmetric baseline system and a structurally perturbed system across parameter spaces. It performs fixed-point detection, stability analysis, parameter sweeps, fine scans, and basin-of-attraction analysis.

All outputs required for the Results section are automatically generated and saved to the Desktop. These include:

Parameter sweep summaries (CSV files)
Fixed-point coordinates
Threshold values across parameter planes
Basin classification results
Figures corresponding to all reported results

To run the analysis:

Open a terminal
Navigate to the script directory
Execute: python3 main.py

The code will create an output directory on the Desktop and export all numerical results and figures used in the manuscript.# gene-revised
