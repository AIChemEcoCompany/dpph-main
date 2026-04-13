# dpph-main
calculation of broken and formed bond for dpph reaction 

## Installation
The main packages can be installed in `.yaml`
- Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- Create conda environment 
```
   conda env create -f dpph_env.yaml 
   conda activate dpph
```
  
## Start up
1. Verify the chemical reaction string and atom mapping.
```
python 1preprocessing_data.py
```
2. Construction chemical bonds.
```
python 2construct_fg.py
```
3. Calculating the bonds of chemical reactions based on functional groups.
```
python 3get_broken.py
```
4. 