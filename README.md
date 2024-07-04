# History matching tutorial

![Screenshots](./screenshots.png)

## Run in the cloud (no installation required)

- on Colab (requires Google login):
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/patnr/HistoryMatching)
- on a NORCE server (not generally available):
  [![JupyterHub](https://img.shields.io/static/v1?label=JupyterHub&message=by%20DIGIRES&logo=jupyter&color=blue)](https://jupyterhub.fredagsmorgen.no/hub?next=%2Fuser-redirect%2Fgit-pull?repo%3Dhttps%253A%252F%252Fgithub.com%252Fpatricknraanes%252FHistoryMatching%26branch%3Dmaster)

## OR: install

Use this option for development, or if you simply want faster computations
(your typical laptop is 10x faster than Google's free offering).

#### Prerequisite: Python>=3.10

If you're an expert, setup a python environment however you like.
Otherwise:
Install [Anaconda](https://www.anaconda.com/download), then
open the [Anaconda terminal](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda)
and run the following commands:

```bash
conda create --yes --name my-env python=3.10
conda activate my-env
python --version
```

Ensure the printed version is 3.10 or higher.  
*Keep using the same terminal for the commands below.*

#### Install

- `git clone` this repository (see the green button up top).  
  *You could instead download & unzip, but then you will
  have to manually download any later updates.*
- Move the resulting folder wherever you like
- `cd` into the folder
- Install requirements:  
  `pip install -r [path/to/]requirements-dev.txt`

#### Launch

- Launch the "notebook server" by executing:  
  `jupyter notebook`  
  This will open up a page in your web browser that is a file navigator.  
- Click on `HistoryMatch.ipynb`.

## Developer guide

I prefer to develop mostly in the format of standard python script,
which is why each notebook corresponds to a `.py` file synced via [jupytext](https://jupytext.readthedocs.io/en/latest/).
The synchronization is done whenever the notebook is saved.
Also, if you run `pre-commit install`,
then the notebooks will get synced with the `.py` files before committing.

Linting (which is, as of now, just a suggestion) can be run with
`ruff check --output-format=grouped`.

## Contributors

This work has been developed by *Patrick N. Raanes*, researcher at *NORCE*.
The project has been funded by *DIGIRES*,
a project sponsored by industry partners
and the *PETROMAKS2* programme of the *Research Council of Norway*.

<a href="http://norceresearch.no">
<img height="100" src="https://github.com/nansencenter/DAPPER/blob/master/docs/imgs/norce-logo.png">
</a>

<a href="http://digires.no">
<img src="http://digires.no/DIGIRES/digilogo%20(002).png" height="100">
</a>

<a href="https://www.data-assimilation.no/projects/remedy">
<img src="./remedy.png?raw=true" height="60">
</a>





<!-- markdownlint-configure-file
{
  "header-increment": false,
  "no-multiple-blanks": false,
  "no-inline-html": {
    "allowed_elements": [ "img", "a" ]
  },
  "code-block-style": false,
  "ul-indent": { "indent": 2 }
}
-->
