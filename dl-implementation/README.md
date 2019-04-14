RNA
===
This python module applies deep learning to classify RNA samples on the basis of several markers

Setup
-----
This module is python 3.6 or higher.
* Install the modules (in a virtual environment) from the `requirements.txt` (replace `tensorflow` with `tensorflow-gpu` if wanted).
* Place the data in the data folder.
* Add a yaml file with the following specification in your home directory (see the module `confidence` for more details):
```yaml
# information regarding the data
files:
  # name of the file (in the data dir) with the single measurements
  single: 'single_name.csv'
  # name of the file (in the data dir) with the mixture measurments
  mixture: 'dataset_mixture_ann.csv'
  # separator of the files
  csv_sep: ','

# sample type names which special role
sample_types:
  # sample type names that are blanks
  blanks:
    - 'sample type name'
  # sample type names that should be filtered if specified
  filter:
    - 'sample type name'

# cut-off value for binary prediction (and validation of measurments)
cut_off: 150

columns:
  # name of the column that indicates the type of the measurement
  type: 'type'
  # name of column that indicates which replicate the measurement is
  replicate: 'replicate_value'
  # markers used to check if sample is valid
  validation:
    - 'marker1'
    - 'marker2'
    - 'etc.'
  # markers used for prediction
  prediction:
    - 'marker1'
    - 'marker2'
    - 'etc.'
```

Run
---
To run the module, execute:
```python
python run-all.py -h
``` 
A help doc will pop up, specify the parameters as you like

Assumptions
-----------
* The replicate values increase by 1
* Mixtures types are seperated with a `+`
* Mixtures have no 'new' types (i.e. unseen in the singles)
