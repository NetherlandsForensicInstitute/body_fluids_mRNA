single_cell_types = \
    ('Blood', 'Saliva', 'Vaginal.mucosa', 'Menstrual.secretion',
     'Semen.fertile', 'Semen.sterile', 'Nasal.mucosa', 'Skin')

string2index = {}
index2string = {}
for i, single_cell_type in enumerate(single_cell_types):
    string2index[single_cell_type] = i
    index2string[i] = single_cell_type

string2index['Skin.penile'] = len(single_cell_types)
index2string[len(single_cell_types)]= 'Skin.penile'


marker_names = ['HBB', 'ALAS2', 'CD93', 'HTN3', 'STATH', 'BPIFA1', 'MUC4', 'MYOZ1', 'CYP2B7P1', 'MMP10', 'MMP7',
                'MMP11', 'SEMG1', 'KLK3', 'PRM1', 'RPS4Y1', 'XIST', 'ACTB', '18S-rRNA']