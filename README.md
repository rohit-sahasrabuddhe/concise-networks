# Concise network models of path data
This repository contains the implementation of the framework described in paper:
Sahasrabuddhe, R., Lambiotte, R., and Rosvall, M., 2025. Concise network models of memory dynamics reveal explainable patterns in path data. arXiv preprint [arXiv:2501.08302](https://arxiv.org/abs/2501.08302).

## Requirements

The code was written in Python 3.9 and uses standard libraries such as `numpy`, `pandas`, and `scipy`. It uses the `sklearn` k-means clustering implementation and `joblib` for parallelising.

## General usage

The `airports_example.ipynb` notebook contains a working example.

## Data and reproducibility
The `synthetic_experiments.ipynb` notebook contains the code to reproduce the results in Fig. 2 of the manuscript. 
`airport_trigrams.txt` contains the transit trigrams extracted from flight itineraries. 
The `lazega_data` folder contains the network structure of the Lazega law firm data.


## Credits

The Convex Non-negative Matrix Factorisation at the core of the framework is due to
Ding, C.H., Li, T. and Jordan, M.I., 2008. Convex and semi-nonnegative matrix factorizations. IEEE transactions on pattern analysis and machine intelligence, 32(1), pp.45-55.

Flight itinerary data is maintained by the U.S. Bureau of Transportation Statistics. [https://www.transtats.bts.gov/DataIndex.asp](https://www.transtats.bts.gov/DataIndex.asp)

The Lazega law firm data was collected by Lazega, E., 2001. The collegial phenomenon: The social mechanisms of cooperation among peers in a corporate law partnership. Oxford University Press, USA. We accessed it on Manlio de Domenico's multilayer network repository. [https://manliodedomenico.com/data.php](https://manliodedomenico.com/data.php)