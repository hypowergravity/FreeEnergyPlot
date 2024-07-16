---
title: MMGBSA decomposition tutorial.
author: Sriram Srinivasa Raghavan
date: April 2023
---


# FreeEnergyPlot
This script generates a plot of the per-residue energy decomposition calculated by AMBER's MMPBSA.py tool using MMPBSA_API. MMPBSA.py is a module in the AMBER molecular dynamics simulation software suite used to estimate the free energy of protein-ligand binding. The per-residue energy decomposition breaks down the total binding energy into contributions from each amino acid residue in the protein. The resulting plot provides insight into the key residues involved in the binding interaction.. 

# MMGBSA

This tutorial covers how to generate a plot of the per-residue energy decomposition using AMBER's MMPBSA.py tool. MMPBSA.py is a module in the AMBER molecular dynamics simulation software suite used to estimate the free energy of protein-ligand binding. The per-residue energy decomposition breaks down the total binding energy into contributions from each amino acid residue in the protein. The resulting plot provides insight into the key residues involved in the binding interaction.

MMGBSA is calculated using the [MMGBSA.py](https://pubs.acs.org/doi/10.1021/ct300418h) tool available in Amber. This tutorial provides an example of free energy calculation in the general case.

## Prerequisites

To run the script, you need a prmtop and trajectory file. Here's an example of how to extract frames from a trajectory file:

1. Check the number of frames using the following command:
~~~bash
cpptraj -p sanga1_protein.prmtop -y 10_frames.xtc -tl
~~~

* Extract the first five frames and output them to a new file:
~~~bash
cat>>traj.in<<EOF
trajout 5_frames.xtc xtc onlyframes 1,2-5
EOF
cpptraj -i traj.in -p sanga1_protein.prmtop -y 10_frames.xtc
~~~

#Running MMGBSA calculation

If you need to change the radii sets for various GB models, you can ignore it unless a metal is present in the protein.
~~~bash
parmed -p sanga1_protein.prmtop<<EOF
changeRadii mbondi3
outparm sanga1_protein_mbondi3.prmtop
quit
EOF
~~~
> Simple input for MMGBSA calculation, the script(FreeEnergyPlot.py) was evaluated for the following, idecomp can be any values from 1-4.

```bash
cat>>mmgbsa.in<<EOF
Sample input file for GB

&general
  startframe=1,interval=1,endframe=5,keep_files=2,verbose=2
/
&gb
igb=5, saltcon=0.150,
/
&decomp
idecomp=2, dec_verbose=2,csv_format=1
/
EOF

```

> If normal mode has to be incorpurated it can be done using the following.

```bash
cat>>mmgbsa.in<<EOF
Sample input file for GB

&general
  startframe=1,interval=1,endframe=5,keep_files=2,verbose=2
/
&gb
igb=5, saltcon=0.150,
/
&decomp
idecomp=2, dec_verbose=2,csv_format=1
/
&nmode
   nmstartframe=1, nmendframe=5,
   nminterval=1, nmode_igb=1, nmode_istrng=0.1,
/

EOF

```

## Preparation of Parameter File

To prepare the parameter files for the MMGBSA calculation, you can use the `ante-MMPBSA.py` script. Here's an example command:

~~~bash
ante-MMPBSA.py -p sanga1_protein.prmtop -c sanga1_protein.prmtop -r rec_wild.prmtop -l lig.prmtop --ligand-mask=:304
~~~

Here, we have chosen the residue mask :304 for preparing the parameter file for free energy calculation.

You can confirm the residue name and number of the selected residue mask using the following command:
~~~bash
cpptraj -p sanga1_protein.prmtop --resmask :304 | awk 'BEGIN {OFS="-"} {print $2,$1}'
~~~
Running the MMGBSA Calculation

Finally, you can run the MMPBSA.py calculation using the parameter files generated above, the trajectory file, and any additional input parameters required by the calculation. Here's an example command:

~~~bash
MMPBSA.py -O -i mmgbsa.in -o FINAL_RESULTS_MMPBSA.dat -do FINAL_DECOMP_MMPBSA.dat -eo energies.csv -deo decompose.csv -cp sanga1_protein.prmtop -rp rec_wild.prmtop -lp lig.prmtop -y 5_frames.xtc
~~~
After running the calculation, you can analyze the results. One important result is the per-residue energy decomposition or pair-wise depending on the idecomp value, which can be plotted using the FreeEnergyPlot.py script. Example plots generated by the script are saved in the image folder. Here's an example command to run the script:

~~~bash
python FreeEnergyPlot.py
~~~

For Pair-wise decomposition residue numnber also can be given, example:
~~~bash
python FreeEnergyPlot.py -n 25 26 169 215 272 300
~~~

If the residue name should also be plotted then the following like command has to be provided, where input prmtop and residue number corresponding to protein and ligand has to be given in amber naming syntax, example:

~~~bash
python FreeEnergyPlot.py -r ":1-304" -p sanga1_protein.prmtop  -s 2 -n 25 26 169 215 272 300 
python FreeEnergyPlot.py -r ":1-304" -p sanga1_protein.prmtop -n 25 26 169 215 272 300
~~~
The -s option will shift the residue incase the number are different in pdb and prmtop
The '-n' which specify the residue number which has to be printed should be given in last.
