# pair_style mlbop command

## Syntax
```
pair_style mlbop
```

## Example
```
pair_style hybrid/overlay pace dftd3 pbe 12.0 8.0  
pair_coeff * * pace Cu.yaml Cu 

pair_style hybrid/overlay pace dftd3 pbe 12.0 8.0  
pair_coeff * * pace Cu.yaml Cu 
pair_coeff * * dftd3 param.dftd3 Cu
```

## Description 

The _mlbop_ pair style computes the machine-learning bond-order potential.

## Note
This implementation was tested in LAMMPS 2 Aug 2023 version.  
It may not work in some environments/versions.  
