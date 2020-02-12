# Capital Investment in Assortative Matching with Large Firms
 
 Code for "Corrigendum to capital investment in Assortative Matching with Large Firms".
 
 This code updates the solution to an assortative matching model of [Eeckhout and Kircher (2018, Econometrica)]( https://doi.org/10.3982/ECTA14450) with capital.
 
 Please contact [Cristina Lafuente](https://sites.google.com/view/clafuente/) (Cristina.Lafuente@eui.eu) if you have any question about the code.
 
 ## The files
 
 The `Corrigendum` jupyter notebook explains the changes to the original paper and the results. The python files (`inputs`,`ivp`,`models`,`shooting`,`solvers`,`new_capital`) are used within the notebook. The build on the shooting solver of [Assortative matching with Large firms code](https://github.com/davidrpugh/assortative-matching-large-firms) developed by [David Pugh](https://github.com/davidrpugh) and [Cristina Lafuente](https://github.com/crisla).
 
 The solutions are saved in .csv files for replication. Figure 1 in the Corrigendum is also provided. The solutions of the calibration in Adamopoulos and Restuccia (2014) are also provided for comparison.

 ## Executing the notebook
 
 You need to install a python 3.6 distribution (such as [Anaconda](https://www.anaconda.com/distribution/)) and the packages below for the code to run. To open and run the notebook, type `jupyter notebook` in a terminal/command prompt window. Make sure you are working in the same folder that contains the python files in this repository.

## Package version:
- Python: 3.6.3
- Jupyter: 5.0.0
- Numpy: 1.17.4
- Scipy: 0.19.1
- Sympy: 1.1.1
- Pandas: 0.20.3
- Matplotlib: 3.1.1
- Seaborn: 0.8.0
- Numba: 0.40.1
