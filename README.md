# History matching tutorial

## Run in the cloud

using Colab (requires no installation, but Google login):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/patricknraanes/HistoryMatching/blob/Colab)

## OR: install

Use this option for development, or if you simply want
faster computations (your typical laptop is twice as powerful
as Google's free offering).

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

- Download and unzip (or `git clone`)
  this repository (see the green button up top)
- Move the resulting folder wherever you like, and `cd` into it
- Install requirements:  
  `pip install -r path/to/requirements.txt`

### Launch

- Launch the "notebook server" by executing:  
  `jupyter-notebook`  
  This will open up a page in your web browser that is a file navigator.  
- Click on `MAIN.ipynb`.

## Contributors

This work has been developed by *Patrick N. Raanes*, researcher at *NORCE*.
The project has been funded by *DIGIRES*,
a project sponsored by industry partners
and the *PETROMAKS2* programme of the *Research Council of Norway*.

<a href="http://norceresearch.no">
<img height="100" src="https://norceresearch.s3.amazonaws.com/_1200x630_crop_center-center_none/norcelogo-metatag.jpg">
</a>

<a href="http://digires.no">
<img src="http://digires.no/DIGIRES/digilogo%20(002).png" height="100">
</a>





<!-- markdownlint-configure-file
{
  "no-multiple-blanks": false,
  "no-inline-html": {
    "allowed_elements": [ "img", "a" ]
  },
  "code-block-style": false,
  "ul-indent": { "indent": 2 }
}
-->
