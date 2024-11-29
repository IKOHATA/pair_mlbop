# pair_style mlbop command

## Syntax
```
pair_style mlbop
```

## Example
```
pair_style mlbop
pair_coeff * * mlbop C.mlbop C 

pair_style hybrid/overlay mlbop dftd3 pbe 12.0 8.0  
pair_coeff * * mlbop C.mlbop C 
pair_coeff * * dftd3 param.dftd3 C
```

## Description 

The _mlbop_ pair style computes the machine-learning bond-order potential. 
This pair style can be used with the pair_style _dftd3_ for the dispersion correction.

## Note
This implementation was tested in LAMMPS 2 Aug 2023 version.  
It may not work in some environments/versions.  
