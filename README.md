# Calculating LRs for presence of body fluids from mRNA assay data in mixtures

This repository contains all data and code needed to reproduce the accompanying 
paper (Ypma et al.).
The data are mRNA measurements collected from samples donated by volunteers. 
We used these construct and evaluate an LR system, that will compute a likelihood ratio
for a pair of hypotheses on the body fluids present in a measured sample, 
as described in Ypma et al. If you find the software or data useful, please cite


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

There is additional code for experiments that did not make the paper. Most notable this includes a
deep learning model (in 'dl-implementation'). The model achieved comparable performance at much 
higher complexity, and would have required detailed explanations if included in the paper.
## Background levels

To compute LRs we need to have knowledge on background levels, which can depend
on non-disputed facts of the case. These background levels are a form of prior, and referenced as such
in the code. They should not be confused with the prior odds, that depend on 
additional case-specific information and does not lie within the domain of 
the forensic scientist. 

For historic reasons, there are two ways to specify background levels. These should be replaced
by one system in which background levels can be specified per background level per hypothesis.

**background levels as proportions**
Within the data augmentation process, the same number of samples for all possible combinations of body fluids are created.
This means that the model will learn that the probability of each of these combinations occurring is the same for all 
combinations. This is the same as assuming uniform priors. However, in reality this will not be the case. For this reason
we wanted to test whether non-uniform priors would have an significant effect on the likelihood ratios that the
model(s) calculate. This is done via the number of samples created in data augmentation process. Within `run.py`
the variable `priors` may be adjusted in the following way (for simplicity assuming
body fluids are (
['blood', 'saliva', 'nasal mucosa']):
* `[10, 1, 1]` : blood is 10x more likely
* `[1, 7, 1]` : saliva is 7x more likely
* `[1, 10, 10]` : blood is 10x less likely
* `[8, 8, 1]` : nasal mucosa is 8x less likely

Note this is a legacy way of defining priors/background levels and can be improved

**background levels of 1 or 0**
Currently, priors 0 or 1 are handled as one huge exceptional case for penile skin. To enable this, make your priors
one longer (in run.py) and add 'Skin.penile' to the body fluids in constants. Yes, I know.


#### Credits    
This repository is based on the work by Naomi Voerman, which she performed for 
her Master's thesis '_A probabilistic approach to quantify the strength of evidence of presence of body fluids from RNA 
data using a multi-label method_'

