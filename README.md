# ct4101-sloinnte-matching
## Sloinnte: Predicting Surname Equivalence Between Irish and English Forms

CT4101 Machine Learning portfolio project.

**Research question:** Can machine-learning models predict whether two surname forms belong to the same surname cluster in the Gaois Database of Irish-Language Surnames?

## Dataset
Bunachar Sloinnte Gaeilge | Database of Irish-Language Surnames (Gaois, Fiontar & Scoil na Gaeilge, DCU).

- Source: [minimal.xml](https://github.com/gaois/sloinnte/blob/4fd4ac91a195ae8d3f524e0d5a692d2d750e92a6/minimal.xml) (commit `4fd4ac9`), saved as `data/minimal.xml`
- Licence: MIT, © Dublin City University
- Citation: Ó Raghallaigh, B., Měchura, M. B., Ó Fionnagáin, A., & Osborne, S. (2021). Developing the Gaois Linguistic Database of Irish-language Surnames. *Names: A Journal of Onomastics*, 69(1), 10–19. https://doi.org/10.5195/names.2021.2251

## Setup
Python 3.13.3

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Running
    jupyter notebook notebooks/01_data_and_baseline.ipynb
    jupyter notebook notebooks/02_model_comparison.ipynb

Run each with **Kernel → Restart & Run All**. Notebook 02 imports shared functions from `src/sloinnte_utils.py`.



