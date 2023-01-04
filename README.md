# ML-EIT-Graphene
## Description
Software based on the [pyEIT](https://github.com/liubenyuan/pyEIT) package, utilising [pyVISA](https://pyvisa.readthedocs.io/en/latest/) for Automating Electrical Impedance Tomography measurements.
This software was used with a SR860 Standford Research Systems lock-in amplifier and a CYTEC VX/256 switchbox to perform 4-terminal
electrical measurements of a pyrolytic graphene foil sample. This work was used to produce a paper: [Machine Learning Enhanced EIT for 2D materials](https://iopscience.iop.org/article/10.1088/1361-6420/ac7743/meta).

`If you intend to use elements of this software, please do not hesitate to get in touch or open an issue so I can point you in the right direction :) This is old work from my Master's so it is not continually developed, nor has it been cleaned up and organised as best as it can be!`

### Main features
* Machine Learning Adaptive Electrode Selection Algorithm (AESA)
* Automation of 4-terminal measurements via PyVisa Application Programming Interface
* Random and user-specified anomaly generation for use in PyEIT
* GPU acceleration of PyEIT package
* Modification of PyEIT forward mesh generation to specify contact widths and position.

The AESA is the most novel part of this work. It was built to make use of measurement automation and GREIT dynamic EIT reconstruction algorithm,
see the paper for details of how it works.

## Repo layout
### Main folders
* adaptive ESA training CPU - This contains the relevant code to run Bayesian optimisation of the AESA's hyperparameters using simulated anomalies. This has the most RAM expensive parts of the the GPU acceleration removed
but still requires a GPU. This allows for optimisation of more contacts and the higher density meshes required.
* adaptive ESA training GPU - This contains the relevant code to run Bayesian optimisation of the AESA's hyperparameters using simulated anomalies. Contains full GPU acceleration
but is quite expensive to run for anything more than 20 contacts.
* eitVISA with adaptive ESA - The main bulk of the code for operation on simulated data and also the modules used for experimental analysis.

### eitVISA with adaptive ESA
* /anomaly_pkl_files
* /example_data_runs
* /pyeit - Our modifed version of pyEIT
* aesa_complexity_test.py
* aesa_iteration.py - Module that contains functions used to call and operate the AESA.
* anomalies.py - Custom anomaly generation.
* component_control.py - `Experimental setup` script used to control experimental setup via PyVisa Automation
* greit_rec_training_set.py
* lockin_test.py* - `Experimental setup` PyVisa testing
* measurement_optimizer.py 
* meshing.py
* parameter_investigation.py 
* reconstruction_plotting.py
* selection_algorithms.py - `Experimental setup`
