# How to validate ML potentials

Some tutorial-style examples for validating machine-learned interatomic potentials, to accompany the article:

<div align="center">

> **[How to validate machine-learned interatomic potentials](https://arxiv.org/abs/2211.12484)**\
> _[Joe D. Morrow](https://twitter.com/JoeMorrow3594), [John L. A. Gardner](https://twitter.com/jla_gardner), and [Volker L. Deringer](http://deringer.chem.ox.ac.uk)_

</div>
 
 
# Files
1. [`demo-error-metrics.ipynb`](demo-error-metrics.ipynb): A Python implementation of the RMSE and MAE metrics described in Fig 3. of the [article](https://arxiv.org/abs/2211.12484)

2. [`demo-error-scaling.ipynb`](demo-error-scaling.ipynb): A notebook to analyse the error scaling of the RMSE per atom with system size, as in Fig. 4

3. [`demo-rotation-invariance.ipynb`](demo-rotation-invariance.ipynb): A demonstration of the dependence of force component MAEs on the orientation of the system

4. [`demo-similarity.ipynb`](demo-similarity.ipynb): A notebook that uses the SOAP kernel to analyse a 10,000-atom compression MD simulation of silicon, as in Fig. 7

5. [`demo-rss.ipynb`](demo-rss.ipynb): A notebook that creates initial 'sensible' random structures (using the `buildcell` code) and generates the plots from Fig. 8.

# Installation

The notebooks above can be run directly, but they may require the installation of some external software if it is not yet available on your system.

To obtain `quippy` (for SOAP) and other dependencies, run

        pip install -r requirements.txt

The code required to run SOAP analysis is free to use for academic purposes (see https://github.com/libAtoms/QUIP for details).

To use `demo-rss.ipynb` completely, it is necessary to install the [`buildcell`](https://www.mtg.msm.cam.ac.uk/files/airss-0.9.1.tgz) code (see https://www.mtg.msm.cam.ac.uk/Codes/AIRSS for details).

# Key references

* The SOAP-similarity-based validation of potentials, on which some of the tutorial examples are based, is described in J. D. Morrow, V. L. Deringer, [J. Chem. Phys. **157**, 104105 (2022)](https://doi.org/10.1063/5.0099929).

* The SOAP kernel itself is described in A. P. Bart&oacute;k, R. Kondor, G. Cs&aacute;nyi, [Phys. Rev. B **87**, 184115 (2013)](https://doi.org/10.1103/PhysRevB.87.184115).

* The AIRSS approach, used in `demo-rss.ipynb`, is described in C. J. Pickard, R. J. Needs, [J. Phys.: Condens. Matter **23** 053201 (2011)](https://doi.org/10.1088/0953-8984/23/5/053201)

* Carbon structural data used for demonstrating error metrics is described in [Phys. Rev. B **95**, 094203](https://doi.org/10.1103/PhysRevB.95.094203).

