# dpph-main

Calculation of broken and formed bonds for DPPH \((aminooxy)diphenylphosphine oxide\) reactions, including functional-group-aware bond analysis, PostgreSQL-based compound availability lookup, and novelty sorting of unbroken bond.

## Table of Contents
- [dpph-main](#dpph-main)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Installation](#installation)
  - [Workflow Overview](#workflow-overview)


## Background
DPPH is a commonly used in antioxidant activity assays. This toolkit processes DPPH‑related reaction data to:
- Extract reaction SMARTS and validate atom mapping.
- Identify functional groups affected by bond breaking/forming.
- Match broken bonds against a database of commercially available molecules.
- Rank novel (previously unobserved) bond cleavages for further exploration.

The pipeline combines RDKit for cheminformatics operations and PostgreSQL for fast substructure matching.

## Installation
We recommend using Miniconda to manage the environment.

1. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main).
2. Create and activate the conda environment:
```bash
git clone https://github.com/AIChemEcoCompany/dpph-main.git
conda env create -f dpph_env.yaml
conda activate dpph
```


## Workflow Overview
The analysis follows five sequential steps, each implemented as a standalone Python script:
|  Step   | Script  | Purpose |
|  ----  | ----  |  ----  |
| 1  | 1preprocessing_data.py | Verify reaction SMILES/SMARTS, ensure correct atom mapping.
| 2  | 2construct_fg.py | Define SMARTS patterns for functional groups of interest.
| 3  | 3get_broken.py | Compute bonds broken/formed during reactions, annotate with functional groups.
| 4  | 4get_avail_mp.py | Load a compound inventory into PostgreSQL and match broken-bond substructures.
| 5  | 5novelty.py | Identify and rank broken bonds that have not been seen before.

A typical run executes all scripts in order.



