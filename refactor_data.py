"""

Adds an extra column with replicate numbers to a dataframe.

"""

import os
import csv
import pandas as pd

from collections import OrderedDict

def getNumeric(prompt):
    while True:
        response = input(prompt)
        try:
            return int(response)
        except ValueError:
            print("Please enter a number.")

def manually_refactor_indices():
    """
    Takes the single cell type dataset with two sheets of which one is metadata.
    From the information the replicate numbers are retrieved and an extra column
    is added to the dataframe. If it is not clear to which replicate  sample
    belongs, in a python shell an integer can be set manually.

    :return: excel file with a column "replicate_values" added
    """

    xls = pd.ExcelFile('Datasets/Dataset_NFI_adj.xlsx')
    sheet_to_df_map = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
    relevant_column = list(zip(*list(sheet_to_df_map['Samples + details'].index)))[3]
    shortened_names = [relevant_column[i][-3:] for i in range(len(relevant_column))]
    replicate_values = OrderedDict([])
    indexes_to_be_checked = []
    for i in range(len(shortened_names)):
        try:
            replicate_values[i] = int(shortened_names[i][-1])
        except ValueError:
            indexes_to_be_checked.append(i)
            replicate_values[i] = shortened_names[i]
        if shortened_names[i] in ['0.3', '.75', '0.5', 'a_1', 'n_1', 'n_4']:
            indexes_to_be_checked.append(i)
            replicate_values[i] = shortened_names[i]
        if shortened_names[i][-1] in ['0', '6', '7']:
            indexes_to_be_checked.append(i)
            replicate_values[i] = shortened_names[i]

    # Iterate over the index and manually adjust
    replace_replicate = []
    for i in range(len(indexes_to_be_checked)):
        print("The middle rowname of the following rownames should get assigned a replicate number:",
              relevant_column[indexes_to_be_checked[i] - 1:indexes_to_be_checked[i] + 2])
        replace_replicate.append(getNumeric("Give the replicate number"))
    for i in range(len(indexes_to_be_checked)):
        replicate_values[indexes_to_be_checked[i]] = replace_replicate[i]
    replicates = list(replicate_values.values())
    replicates.insert(0, "replicate_value")
    csvfile = "Datasets/replicate_numbers_single.csv"
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in replicates:
            writer.writerow([val])

    sheet_to_df_map['Samples + details'].to_csv('Datasets/Dataset_NFI_meta.csv', encoding='utf-8')
    sheet_to_df_map['Data uitgekleed'].to_csv('Datasets/Dataset_NFI_adj.csv', encoding='utf-8')
    a = pd.read_csv("Datasets/Dataset_NFI_meta.csv", delimiter=',')
    b = pd.read_csv("Datasets/Dataset_NFI_adj.csv", delimiter=',')
    c = pd.read_csv("Datasets/replicate_numbers_single.csv")
    mergedac = pd.concat([a, c], axis=1)
    for i in range(len(mergedac.columns.values)):
        if 'Unnamed' in mergedac.columns.values[i]:
            mergedac.columns.values[i] = None
    mergedbc = pd.concat([b, c], axis=1)
    for i in range(len(mergedbc.columns.values)):
        if 'Unnamed' in mergedbc.columns.values[i]:
            mergedbc.columns.values[i] = None
    mergedac.to_excel("Datasets/Dataset_NFI_meta_rv.xlsx", index=False)
    mergedbc.to_excel("Datasets/Dataset_NFI_rv.xlsx", index=False)

    os.remove("Datasets/Dataset_NFI_meta.csv")
    os.remove("Datasets/Dataset_NFI_adj.csv")
    os.remove("Datasets/replicate_numbers_single.csv")

def manually_refactor_indices_mixtures():
    """
    Takes the mixture cell type dataset with two sheets of which one is metadata.
    From the information the replicate numbers are retrieved and an extra column
    is added to the dataframe. If it is not clear to which replicate  sample
    belongs, in a python shell an integer can be set manually.

    :return: excel file with a column "replicate_values" added
    """
    xls = pd.ExcelFile('Datasets/Dataset_mixtures_adj.xlsx')
    sheet_to_df_map = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
    relevant_column = list(zip(*list(sheet_to_df_map['Mix + details'].index)))[3]
    shortened_names = [relevant_column[i][-3:] for i in range(len(relevant_column))]
    replicate_values = OrderedDict([])

    for i in range(len(shortened_names)):
        try:
            replicate_values[i] = int(shortened_names[i][-1])
        except ValueError:
            print("Some samples do not have clear replicate numbers")
    replicates = list(replicate_values.values())
    replicates.insert(0, "replicate_value")
    csvfile = "Datasets/replicate_numbers_mixture.csv"

    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in replicates:
            writer.writerow([val])

    sheet_to_df_map['Mix + details'].to_csv('Datasets/Dataset_mixture_meta.csv', encoding='utf-8')
    sheet_to_df_map['Mix data uitgekleed'].to_csv('Datasets/Dataset_mixture_adj.csv', encoding='utf-8')
    a = pd.read_csv("Datasets/Dataset_mixture_meta.csv", delimiter=',')
    b = pd.read_csv("Datasets/Dataset_mixture_adj.csv", delimiter=',')
    c = pd.read_csv("Datasets/replicate_numbers_mixture.csv")
    mergedac = pd.concat([a, c], axis=1)
    for i in range(len(mergedac.columns.values)):
        if 'Unnamed' in mergedac.columns.values[i]:
            mergedac.columns.values[i] = None
    mergedbc = pd.concat([b, c], axis=1)
    for i in range(len(mergedbc.columns.values)):
        if 'Unnamed' in mergedbc.columns.values[i]:
            mergedbc.columns.values[i] = None
    mergedac.to_excel("Datasets/Dataset_mixtures_meta_rv.xlsx", index=False)
    mergedbc.to_excel("Datasets/Dataset_mixtures_rv.xlsx", index=False)

    os.remove("Datasets/Dataset_mixture_meta.csv")
    os.remove("Datasets/Dataset_mixture_adj.csv")
    os.remove("Datasets/replicate_numbers_mixture.csv")