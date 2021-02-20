# Contributing

It is great to hear you are interested in contributing to the project!

First, you will need to fork this repository and make a branch for your edits.
*bajes* is currently under development and the repository could be updated 
while you are editing your branch.
For more information, please contact the [maintainer](mailto:matteo.breschi@uni-jena.de?subject=[GitHub]%20Bajes%20contributing).
Once you have made and tested your changes, 
push and submit a pull request, filling out the necessary information. 

For the development of the software, 
it is highly recommended to follow the logic of the implemented modular architecture and the general guidelines:
* The `inf` module represents an implementation of the Bayesian logic and it is the skeleton of the inference routines. 
This module should contains only Bayesian methods and techniques aiming to generality, versatility and braod applicability.
* The `obs` module containes tools for the physical characterization of the observational data with the purpose of defining a full Bayesian model.
Each kind of physical model should be fully contained in a related sub-module, such as `gw` and `kn`.
* It is important to keep track of the relevant changes of the software in the [`CHANGELOG`](CHANGELOG.md).
If you are going to implement major changes, please update it with the relevant information.
