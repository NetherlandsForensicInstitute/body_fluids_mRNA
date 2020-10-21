# Calculating LRs for presence of body fluids from mRNA assay data in mixtures

This repository contains the data and scripts used for ... If you find the software or data useful, please cite


    @article{ypma,
	Author = {RJF Ypma and P Maaskant and R Gill and M Sjerps and M van den Berge},
	Title = {Calculating LRs for presence of body fluids from mRNA assay data in mixtures},
    }

## Contents
Running `run.py` will generate all (data-based) figures in the accompanying article.

The actual work is done in `analytics.py` and `analysis.py`.

## Background levels

There are two types of priors about which we are concerned. The first type refers to what we expect to see when only
taking non-disputed facts of the case into consideration. We refer to this in the paper as 'background levels'.
The second type refers to additional information in the case relating to guilt, and does not lie within the domain of 
the forensic scientist. This type of prior is not referenced anywhere in the code.

For historic reasons, there are two ways to specify background levels.

**background levels as proportions**
Within the data augmentation process, the same number of samples for all possible combinations of cell types are created.
This means that the model will learn that the probability of each of these combinations occurring is the same for all 
combinations. This is the same as assuming uniform priors. However, in reality this will not be the case. For this reason
we wanted to test whether non-uniform priors would have an significant effect on the likelihood ratios that the
model(s) calculate. This is done via the number of samples created in data augmentation process. Within `run.py`
the variable `priors` may be adjusted in the following way:

To simplify I assume that there are three cell types in total.
['blood', 'saliva', 'nasal mucosa']
* `[10, 1, 1]` : blood is 10x more likely
* `[1, 7, 1]` : saliva is 7x more likely
* `[1, 10, 10]` : blood is 10x less likely
* `[8, 8, 1]` : nasal mucosa is 8x less likely

Note this is a legacy way of defining priors/background levels and can be improved

**background levels of 1 or 0**
Currently, priors 0 or 1 are handled as one huge exceptional case for penile skin. To enable this, make your priors
one longer (in run.py) and add 'Skin.penile' to the cell types in constants. Yes, I know.


####Credits    
This repository is based on the work by Naomi Voerman, which she performed for 
her Master's thesis '_A probabilistic approach to quantify the strength of evidence of presence of cell types from RNA 
data using a multi-label method_'

