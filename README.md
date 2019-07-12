## Priors

There are two types of priors about which we are concerned. The first 'type' is mostly important within the experiments. 
The second 'type' is important within case work.

1. **priors within experiments**

Defined as `priors`.

Within the data augmentation process, the same number of samples for all possible combinations of cell types are created.
This means that the model will learn that the probability of each of these combinations occurring is the same for all 
combinations. This is the same as assuming uniform priors. However, in reality this will not be the case. For this reason
we wanted to test whether non-uniform priors would have an significant effect on the likelihood ratios that the
model(s) calculate. Via the number of samples created in data augmentation process this is encountered. Within `settings.py`
the variable `priors` may be adjusted in the following way:

To simplify I assume that there are three cell types in total.
['blood', 'saliva', 'nasal mucosa']
* `[10, 1, 1]` : blood is 10x more likely
* `[1, 7, 1]` : saliva is 7x more likely
* `[1, 10, 10]` : blood is 10x less likely
* `[8, 8, 1]` : nasal mucosa is 8x less likely

2. **priors within case work**

Defined as `priors_numerator` and  `priors_denominator`.
