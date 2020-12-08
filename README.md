# History matching tutorial

<img src="https://norceresearch.s3.amazonaws.com/_1200x630_crop_center-center_none/norcelogo-metatag.jpg" height="200">

## Run in the cloud

using Colab (requires no installation, but Google login):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/patricknraanes/HistoryMatching/blob/Colab)

## OR: install

This option should be preferred for development,
or just because your local computer is usually faster than what Google will provide you for free.

#### Prerequisite: Python>=3.7

If you're not an admin or expert:  

- Install [Anaconda](https://www.anaconda.com/download).
- Open the [Anaconda terminal](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda)
  and run the following commands:

      conda create --yes --name my-env python=3.8
      conda activate my-env
      python -c 'import sys; print("Version:", sys.version.split()[0])'

  Ensure the output at the end gives a version bigger than 3.7.  
  Keep using the same terminal for the commands below. 

#### Install

- Download and unzip (or `git clone`) this repository.
- Move the resulting folder wherever you like,  
  and `cd` into it
- Install requirements:
  `pip install -r path/to/requirements.txt`.
- Launch Jupyter:  
  `jupyter-notebook`.  
  This will open up a page in your web browser that is a file navigator.  
  Click on `MAIN.ipynb`.
