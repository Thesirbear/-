## Getting started

### Code and development environment

Right now this project can be deployed on machines with cuda and any other on cpu(Not recommended due to big duration of training), model originally was trained on kaggle with 2xT4, and it took only 90-120 minutes.

> Install `conda` and create virtual environment, then run in conda-osx-arm64-mps `conda env export -n <your_env_name> -f environment.yml` command, this will create exact same env.
Then go to reproducibility-scripts section and run `train.sh` with `zsh` or `sh` command, you will need to login to your W&B account.



## Licenses and acknowledgements

This project is licensed under the LICENSE file in the root directory of the project.

The initial code of this repository has been initiated by the [Python Machine Learning Research Template](https://github.com/CLAIRE-Labo/python-ml-research-template)
with the LICENSE.ml-template file.



Used materials:

[How to write Relu from Scratch](https://www.youtube.com/watch?v=93qjwrP7PfE) - Used this one in order to understand how to implement custom activational functions, then i use some of this information to build mfm (Max-feature-map) activation function, to run LCNN.
