import numpy as np



# TODO: Remove last 4 marker names?
marker_names = ['HBB', 'ALAS2', 'CD93', 'HTN3', 'STATH', 'BPIFA1', 'MUC4', 'MYOZ1', 'CYP2B7P1', 'MMP10', 'MMP7',
                'MMP11', 'SEMG1', 'KLK3', 'PRM1', 'RPS4Y1', 'XIST', 'ACTB', '18S-rRNA']

def make_nhot_matrix_of_combinations(N):
    """
    Makes nhot encoded matrix with all possible combinations of existing
    single cell types.

    :param N: int
    :return: 2**N x N nhot encoded matrix
    """

    def int_to_binary(i):
        binary = bin(i)[2:]
        while len(binary) < N:
            binary = '0' + binary
        return np.flip([int(j) for j in binary]).tolist()

    return np.array([int_to_binary(i) for i in range(2**N)])

# nhot_matrix_all_combinations = make_nhot_matrix_of_combinations(len(single_cell_types))

celltype_specific_markers = dict()
celltype_specific_markers['Blood'] = ['HBB', 'ALAS2', 'CD93']
celltype_specific_markers['Saliva'] = ['HTN3', 'STATH']
celltype_specific_markers['Vaginal.mucosa'] = ['MUC4', 'MYOZ1', 'CYP2B7P1']
celltype_specific_markers['Menstrual.secretion'] = ['MMP10', 'MMP7', 'MMP11']
celltype_specific_markers['Semen.fertile'] = ['SEMG1', 'KLK3', 'PRM1']
celltype_specific_markers['Semen.sterile'] = ['SEMG1', 'KLK3', 'PRM1']
celltype_specific_markers['Nasal.mucosa'] = ['STATH', 'BPIFA1']
celltype_specific_markers['Skin'] = [None]

model_names = {
    'Vaginal mucosa and/or Menstrual secretion no Skin Penile': 'vagmenstr_no_penile',
    'Vaginal mucosa and/or Menstrual secretion Skin Penile': 'vagmenstr_penile',
    'Saliva no Skin Penile': 'saliva_no_penile',
    'Saliva Skin Penile': 'saliva_penile'
}