# History matching tutorial

## Jump right in
using Google colab (requires login):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/patricknraanes/HistoryMatching/blob/Colab)

## Installation

### Prerequisite: Python>=3.7

If you're not an admin or expert:  

- Install [Anaconda](https://www.anaconda.com/download).
- Open the [Anaconda terminal](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda)
  and run the following commands:

      conda create --yes --name my-env python=3.8
      conda activate my-env
      python -c 'import sys; print("Version:", sys.version.split()[0])'

  Ensure the output at the end gives a version bigger than 3.7.  
  Keep using the same terminal for the commands below. 

### Install

- Download and unzip (or `git clone`) this repository.
- Move the resulting folder wherever you like,  
  and `cd` into it
- Install requirements:
  `pip install -r path/to/requirements.txt`.
- Launch Jupyter:  
  `jupyter-notebook`.  
  This will open up a page in your web browser that is a file navigator.  
  Click on a `notebook1.py`.  
  (The jupyter extension `jupytext` will convert it to the notebook format)
