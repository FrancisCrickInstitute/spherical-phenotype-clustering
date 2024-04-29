# Spherical Phenotype Clustering


Run the code with `python main.py <config filepath>`. For example 
`python main.py config/default.yml`. The output will be saved under
an `experiments` folder, in a subdirectory with the same name as the
config file. The config file `config/default.yml` has a hyperparameter
configuration that we have found usually gives reasonable results.

We use a particular format for the CSV referenced in the configuration files.
Required  columns can be found in `dfconst.py`. If you use a different set
of column names for  your dataframes, it should be straightforward to use
them by modifying `dfconst.py`.

## Preprint

Spherical Phenotype Clustering. Luke Nightingale, Joseph Tuersley, Andrea Cairoli, Jacob Howes, Andrew Powell, Darren Green, Amy Strange, Scott Warchal, Michael Howell.
bioRxiv 2024.04.19.590313; doi: https://doi.org/10.1101/2024.04.19.590313
