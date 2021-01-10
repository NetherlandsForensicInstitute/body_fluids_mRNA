# Calculating LRs for presence of body fluids from mRNA assay data in mixtures

This repository contains all data and code needed to reproduce the accompanying 
paper (Ypma et al.).
The data are mRNA measurements collected from samples donated by volunteers. 
We used these construct and evaluate an LR system, that will compute a likelihood ratio
for a pair of hypotheses on the body fluids present in a measured sample, 
as described in the paper. If you find the software or data useful, please cite


    @article{ypma,
	Author = {RJF Ypma and P Maaskant-van Wijk and R Gill and M Sjerps and M van den Berge},
	Title = {Calculating LRs for presence of body fluids from mRNA assay data in mixtures},
    Journal = {Forensic Science International: Genetics}
}

## Technical notes
The datasets should be self explanatory. Note that the column 'replicate_value' specifies
which replicates belong together - replicates from the same sample are numbered consecutively.

Running `run.py` will generate all (data-based) figures in the accompanying article. 
The actual work is done in `analytics.py` and `analysis.py`.
Results are written to the folders 'output' and 'final_model'.

There is additional code for experiments that did not make the paper. Most notably this includes a
deep learning model (in 'dl-implementation'). This model achieved comparable performance at much 
higher complexity, and would have required detailed explanations if included in the paper. 
Furthermore, we looked at a situation where the defense proposes an alternative (e.g. 
sample contains blood, not menstrual secretion), which changed results little.

To compute LRs we need to have knowledge on background levels, which can depend
on non-disputed facts of the case. These background levels are a form of prior, and referenced as such
in the code. They should not be confused with the prior odds, that depend on 
additional case-specific information and do not lie within the domain of 
the forensic scientist'. 
For historic reasons, there are two limited ways to specify background levels,
either through the 'priors_list' or using the flag 'from_penile'. In future work, these should be replaced
by one system in which background levels can be specified per body fluid per hypothesis.

#### Credits    
This repository is based on the work by Naomi Voerman, which she performed for 
her Master's thesis '[_A probabilistic approach to quantify the strength of evidence of presence of body fluids from RNA 
data using a multi-label method_][1]. 

[1]: https://www.universiteitleiden.nl/binaries/content/assets/science/mi/scripties/statscience/2019-2020/finalversion_masterthesis_naomivoerman_s2072661.pdf

