import os
import pickle
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

import numpy as np
import pandas as pd


# Create a window that fills the screen.
from input_output import read_df
from analytics import combine_samples

from rna.input_output import get_data_per_cell_type


class FullScreenApp(object):

    def __init__(self, master):
        self.master = master
        pad = 3
        self._geom = '200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))


class AnalyseExcel:

    def __init__(self):
        self.single_cell_types = ['Semen.fertile', 'Saliva', 'Nasal.mucosa', 'Menstrual.secretion', 'Blood',
                                  'Semen.sterile', 'Vaginal.mucosa', 'Skin']

        self.master = Tk()
        self.master.minsize(1500, 400)
        self.master.geometry("320x100")

        self.open_filename = '../Datasets/example_case.xlsx'
        self.save_filename = '../scratch/test'
        self.save_text_widget = None
        self.open_text_widget = None
        self.number_of_replicates = 4

        self.is_penile = BooleanVar()

        # cell class name to {Uniform, 1, 0} prior
        self.top_variables = {}
        self.bottom_variables = {}

        self.add_buttons()

        self.master.grid_rowconfigure(15, minsize=50)

    def run(self):
        mainloop()

        ################################################################################

        self.analyse_data()

    # Function that saves the location of the input file.
    def open_file_name(self):
        self.open_filename = filedialog.askopenfilename(filetypes=(("Excel files", "*.xlsx")
                                                                   , ("All files", "*.*")))
        self.open_text_widget.delete('1.0', END)
        self.open_text_widget.insert(END, self.open_filename)

    # Function that saves the selected location for the results file.
    def save_file_location(self):
        self.save_filename = filedialog.asksaveasfilename(initialdir="/home/stagiair/Desktop/",
                                                          filetypes=(("CSV files", "*.csv")
                                                                     , ("All files", "*.*")))
        self.save_text_widget.delete('1.0', END)
        self.save_text_widget.insert(END, self.save_filename)

    # Function to restart the program.
    def restart_program(self):
        python = sys.executable
        os.execl(python, python, *sys.argv)

    # Function that checks if the input file and the save location have been selected. If not, a text box will appear.
    def close(self):
        if self.open_filename and self.save_filename:
            self.master.destroy()
        # if open_file and not save_file:
        #     t = Text(master, height=2, width=50)
        #     t.grid(row=3, column=3)
        #     t.insert(END, 'Please select a location to save the results')
        # if not open_file and save_file:
        #     t = Text(master, height=2, width=50)
        #     t.grid(row=2, column=3)
        #     t.insert(END, 'Please select a file to open')
        # if not open_file and not save_file:
        #     t = Text(master, height=2, width=50)
        #     t.grid(row=2, column=3)
        #     t.insert(END, 'Please select a file to open')
        #     t_1 = Text(master, height=2, width=50)
        #     t_1.grid(row=3, column=3)
        #     t_1.insert(END, 'Please select a location to save the results')

    # Function that creates the LR results table
    def create_table(self, cell_types):
        for i in range(len(cell_types)):
            self.tree.heading(1, text="Sample name")
            self.tree.column(1, width=150)

            self.tree.heading(i + 2, text=cell_types[i])
            self.tree.column(i + 2, width=150)

    # Function that gives the option to set the number of replicates
    def set_replicate_number(self):
        top = Toplevel(self.master)
        top.title('Set number of replicates')

        Label(top, text="Set number of replicates").pack()
        e_1 = Entry(top)
        e_1.insert(0, self.number_of_replicates)
        e_1.pack(padx=5)

        def ok():
            self.number_of_replicates = e_1.get()

            top.destroy()

        b = Button(top, text="OK", command=ok)
        b.pack(pady=5)

    def analyse_data(self):
        # global master, self.tree, button_load

        model_filename, marker_names, names, X_single, n_celltypes_with_penile, \
            n_features, n_per_celltype, string2index, index2string = self.load_data()

        X = combine_samples(X_single)

        print('data loaded, shape {}. {}'.format(X.shape, X[0, :]))

        column_names = ['HBB', 'ALAS2', 'CD93', 'HTN3', 'STATH', 'BPIFA1', 'MUC4', 'MYOZ1', 'CYP2B7P1', 'MMP10', 'MMP7',
                        'MMP11', 'SEMG1', 'KLK3', 'PRM1', 'ACTB', '18S-rRNA']
        n_single_cell_types=8

        test_data_grouped = []
        predicted_proba_average = []
        predicted_proba_4 = []
        proba_final_top = []
        proba_final_bottom = []
        if marker_names != column_names:
            messagebox.showinfo("Warning",
                                "'The marker labels are inconsistent with the trained model, please fix the labels. "
                                "The correct labels are: {}. Found {}".format(column_names, marker_names))

        # Load the trained model and all classes present in the trained model.
        model = pickle.load(open(model_filename, 'rb'))

        priors=None # TODO
        lrs =  model.predict_lrs(X,[1]*len(self.single_cell_types), priors=priors)
        #
        # # classes = pickle.load(open('classes.pkl', 'rb'))
        # # mixture_classes_in_single_cell_type = pickle.load(open('mixture_classes_in_single_cell_type', 'rb'))
        # prob_per_class = get_prob_per_class(X, mixture_classes_in_single_cell_type, model, max_lr=10)
        #
        # print(prob_per_class)
        # print(prob_per_class.shape)
        # # Predict the probabilities for the input data for every trained class.
        # predict_proba = model.predict_lrs(X)
        # # predict_proba = predict_proba.toarray()
        #
        # predicted_proba_4.append(predict_proba)
        # # predicted_proba_average.append(sum(predict_proba) / self.number_of_replicates)
        #
        # proba_list = []
        # LR_prediction_list = []
        # top_list = []
        # bottom_list = []
        # final_list = []
        #
        # # all_cell_types = ['Blank_PCR', 'Semen.fertile', 'Saliva', 'Nasal.mucosa', 'Menstrual.secretion', 'Blood',
        # #                   'Semen.sterile', 'Vaginal.mucosa', 'Skin', 'Skin.penile']
        #
        # cell_types_yes_top = [self.single_cell_types[i] for i in
        #                       [i for i, x in enumerate(self.top_variables) if x == 'Always']]
        # cell_types_no_top = [self.single_cell_types[i] for i in
        #                      [i for i, x in enumerate(self.top_variables) if x == 'Never']]
        #
        # cell_types_yes_bottom = [self.single_cell_types[i] for i in
        #                          [i for i, x in enumerate(self.bottom_variables) if x == 'Always']]
        # cell_types_no_bottom = [self.single_cell_types[i] for i in
        #                         [i for i, x in enumerate(self.bottom_variables) if x == 'NEVER']]
        #
        # # TOP PART OF LR
        # for probabilility_4, probability_average in zip(predicted_proba_4, predicted_proba_average):
        #     proba_all_top = []
        #     # Probability for 4 replicates
        #     for probability_single in probabilility_4:
        #         proba_per_class = []
        #         matches_yes_list = []
        #         matches_no_list = []
        #         if len(cell_types_yes_top) != 0:
        #             for single_cell_type in cell_types_yes_top:
        #                 matches_yes = [i for i, s in enumerate(classes) if single_cell_type in s]
        #                 matches_yes_list.append(matches_yes)
        #             flatten_yes = [item for sublist in matches_yes_list for item in sublist]
        #             new_list_yes = sorted(set(flatten_yes))
        #             dup_list_yes = []
        #             for i in range(len(new_list_yes)):
        #                 if (flatten_yes.count(new_list_yes[i]) > len(cell_types_yes_top) - 1):
        #                     dup_list_yes.append(new_list_yes[i])
        #
        #         else:
        #             for single_cell_type in self.single_cell_types:
        #                 matches_yes = [i for i, s in enumerate(classes) if single_cell_type in s]
        #                 matches_yes_list.append(matches_yes)
        #             dup_list_yes = [item for sublist in matches_yes_list for item in sublist]
        #
        #         for single_cell_type_no in cell_types_no_top:
        #             matches_no = [i for i, s in enumerate(classes) if single_cell_type_no in s]
        #             matches_no_list.append(matches_no)
        #         flatten_no = [item for sublist in matches_no_list for item in sublist]
        #
        #         difference_top_list = list(set(list(set(dup_list_yes))) - set(list(set(flatten_no))))
        #
        #         for class_index in difference_top_list:
        #             proba_per_class.append(probability_single[class_index])
        #         proba_all_top.append(sum(proba_per_class))
        #
        #     # Probability for average of 4 replicates
        #     proba_per_class = []
        #     matches_yes_list = []
        #     matches_no_list = []
        #     if len(cell_types_yes_top) != 0:
        #         for single_cell_type in cell_types_yes_top:
        #             matches_yes = [i for i, s in enumerate(classes) if single_cell_type in s]
        #             matches_yes_list.append(matches_yes)
        #         flatten_yes = [item for sublist in matches_yes_list for item in sublist]
        #         new_list_yes = sorted(set(flatten_yes))
        #         dup_list_yes = []
        #         for i in range(len(new_list_yes)):
        #             if (flatten_yes.count(new_list_yes[i]) > len(cell_types_yes_top) - 1):
        #                 dup_list_yes.append(new_list_yes[i])
        #     else:
        #         for single_cell_type in self.single_cell_types:
        #             matches_yes = [i for i, s in enumerate(classes) if single_cell_type in s]
        #             matches_yes_list.append(matches_yes)
        #         dup_list_yes = [item for sublist in matches_yes_list for item in sublist]
        #
        #     for single_cell_type_no in cell_types_no_top:
        #         matches_no = [i for i, s in enumerate(classes) if single_cell_type_no in s]
        #         matches_no_list.append(matches_no)
        #     flatten_no = [item for sublist in matches_no_list for item in sublist]
        #
        #     difference_top_list = list(set(list(set(dup_list_yes))) - set(list(set(flatten_no))))
        #
        #     for class_index in difference_top_list:
        #         proba_per_class.append(probability_average[class_index])
        #     proba_all_top.append(sum(proba_per_class))
        #     proba_final_top.append(proba_all_top)
        #
        # # BOTTOM PART OF LR
        # for probabilility_4, probability_average in zip(predicted_proba_4, predicted_proba_average):
        #     proba_all_bottom = []
        #     # Probability for 4 replicates
        #     for probability_single in probabilility_4:
        #         proba_per_class = []
        #         matches_yes_list = []
        #         matches_no_list = []
        #         if len(cell_types_yes_bottom) != 0:
        #             for single_cell_type in cell_types_yes_bottom:
        #                 matches_yes = [i for i, s in enumerate(classes) if single_cell_type in s]
        #                 matches_yes_list.append(matches_yes)
        #             flatten_yes = [item for sublist in matches_yes_list for item in sublist]
        #             new_list_yes = sorted(set(flatten_yes))
        #             dup_list_yes = []
        #             for i in range(len(new_list_yes)):
        #                 if (flatten_yes.count(new_list_yes[i]) > len(cell_types_yes_bottom) - 1):
        #                     dup_list_yes.append(new_list_yes[i])
        #         else:
        #             for single_cell_type in self.single_cell_types:
        #                 matches_yes = [i for i, s in enumerate(classes) if single_cell_type in s]
        #                 matches_yes_list.append(matches_yes)
        #             dup_list_yes = [item for sublist in matches_yes_list for item in sublist]
        #
        #         for single_cell_type_no in cell_types_no_bottom:
        #             matches_no = [i for i, s in enumerate(classes) if single_cell_type_no in s]
        #             matches_no_list.append(matches_no)
        #         flatten_no = [item for sublist in matches_no_list for item in sublist]
        #
        #         difference_bottom_list = list(set(list(set(dup_list_yes))) - set(list(set(flatten_no))))
        #
        #         for class_index in difference_bottom_list:
        #             proba_per_class.append(probability_single[class_index])
        #         proba_all_bottom.append(sum(proba_per_class))
        #
        #     # Probability for average of 4 replicates
        #     proba_per_class = []
        #     matches_yes_list = []
        #     matches_no_list = []
        #     if len(cell_types_yes_bottom) != 0:
        #         for single_cell_type in cell_types_yes_bottom:
        #             matches_yes = [i for i, s in enumerate(classes) if single_cell_type in s]
        #             matches_yes_list.append(matches_yes)
        #         flatten_yes = [item for sublist in matches_yes_list for item in sublist]
        #         new_list_yes = sorted(set(flatten_yes))
        #         dup_list_yes = []
        #         for i in range(len(new_list_yes)):
        #             if (flatten_yes.count(new_list_yes[i]) > len(cell_types_yes_bottom) - 1):
        #                 dup_list_yes.append(new_list_yes[i])
        #     else:
        #         for single_cell_type in self.single_cell_types:
        #             matches_yes = [i for i, s in enumerate(classes) if single_cell_type in s]
        #             matches_yes_list.append(matches_yes)
        #         dup_list_yes = [item for sublist in matches_yes_list for item in sublist]
        #
        #     for single_cell_type_no in cell_types_no_bottom:
        #         matches_no = [i for i, s in enumerate(classes) if single_cell_type_no in s]
        #         matches_no_list.append(matches_no)
        #     flatten_no = [item for sublist in matches_no_list for item in sublist]
        #
        #     difference_bottom_list = list(set(list(set(dup_list_yes))) - set(list(set(flatten_no))))
        #
        #     for class_index in difference_bottom_list:
        #         proba_per_class.append(probability_average[class_index])
        #     proba_all_bottom.append(sum(proba_per_class))
        #     proba_final_bottom.append(proba_all_bottom)
        #
        # # Calculate the LR
        # for proba_one_top, proba_one_bottom in zip(proba_final_top, proba_final_bottom):
        #     LR_list = []
        #     top_list_temp = []
        #     bottom_list_temp = []
        #     final_list_temp = []
        #     for prob_one_top, prob_one_bottom in zip(proba_one_top, proba_one_bottom):
        #         top_list_temp.append(np.sum(prob_one_top))
        #         bottom_list_temp.append(np.sum(prob_one_bottom))
        #         LR_list.append(np.log10(np.sum(prob_one_top) / np.sum(prob_one_bottom)))
        #         final_list_temp.append([np.sum(prob_one_top), np.sum(prob_one_bottom),
        #                                 np.log10(np.sum(prob_one_top) / np.sum(prob_one_bottom))])
        #
        #     final_list.append(final_list_temp)
        #     top_list.append(top_list_temp)
        #     bottom_list.append(bottom_list_temp)
        #     LR_prediction_list.append(LR_list)

        # # Create a window that shows the output table with the LR's
        # master = Tk()
        # app = FullScreenApp(master)
        #
        # frame = Frame(master)
        # frame.pack()
        #
        # neutral_list_top = [x for x in self.single_cell_types if x not in cell_types_yes_top]
        # neutral_list_top = [x for x in neutral_list_top if x not in cell_types_no_top]
        #
        # neutral_list_bottom = [x for x in self.single_cell_types if x not in cell_types_yes_bottom]
        # neutral_list_bottom = [x for x in neutral_list_bottom if x not in cell_types_no_bottom]
        #
        # # LR table
        # text = Text(frame, width=200, height=1)
        # text.insert('1.0', cell_types_yes_top)
        # text.insert('1.0', 'Top yes: ')
        # text.pack(side=TOP)
        # text1 = Text(frame, width=200, height=1)
        # text1.insert('1.0', cell_types_no_top)
        # text1.insert('1.0', 'Top no: ')
        # text1.pack(side=TOP)
        # text2 = Text(frame, width=200, height=1)
        # text2.insert('1.0', neutral_list_top)
        # text2.insert('1.0', 'Top neutral: ')
        # text2.pack(side=TOP)
        # text3 = Text(frame, width=200, height=1)
        # text3.insert('1.0', cell_types_yes_bottom)
        # text3.insert('1.0', 'Bottom yes: ')
        # text3.pack(side=TOP)
        # text4 = Text(frame, width=200, height=1)
        # text4.insert('1.0', cell_types_no_bottom)
        # text4.insert('1.0', 'Bottom no: ')
        # text4.pack(side=TOP)
        # text5 = Text(frame, width=200, height=1)
        # text5.insert('1.0', neutral_list_bottom)
        # text5.insert('1.0', 'Bottom neutral: ')
        # text5.pack(side=TOP)
        #
        # labels = ['Probability top', 'Probability bottom', 'Log(10) LR']
        # labels_csv = ['Probability top', 'Probability bottom', 'Log(10) LR', 'Top yes', 'Top no', 'Top neutral',
        #               'Bottom no', 'Bottom yes', 'Bottom neutral']
        #
        # number_columns = range(1, (len(labels) + 2))
        #
        # self.tree = ttk.Treeview(frame, columns=number_columns, height=20, show="headings")
        # self.tree.pack(side=TOP)
        #
        # self.create_table(labels)
        #
        # i = 1
        # j = 0
        # values = []
        #
        # temp_list_grouped = []
        # for grouped_LR in final_list:
        #
        #     temp_value = []
        #     for val in grouped_LR:
        #         val = [round(v, 2) for v in val]
        #         if i % (self.number_of_replicates + 1) == 0:
        #             index = 'Average'
        #         else:
        #             index = names[j]
        #             j = j + 1
        #
        #         values.append(index)
        #         if i % (self.number_of_replicates + 1) == 0:
        #             self.tree.insert('', 'end', values=(
        #                 index, val[0], val[1], val[2]), tags=('average',))
        #         else:
        #             self.tree.insert('', 'end', values=(
        #                 index, val[0], val[1], val[2]), tags=('normal',))
        #         i = i + 1
        #         temp_value.append(val)
        #     temp_list_grouped.append(temp_value)
        #
        # self.tree.tag_configure('average', background='lightblue')
        #
        # frames = []
        # for LR_grouped in temp_list_grouped:
        #     df = pd.DataFrame.from_records(LR_grouped, columns=labels)
        #     frames.append(df)
        #
        # # Save the LR selection in a dataframe
        # d = {'Top_yes': [cell_types_yes_top], 'Top_no': [cell_types_no_top], 'Top_neutral': [neutral_list_top],
        #      'Bottom_yes': [cell_types_yes_bottom], 'Bottom_no': [cell_types_no_bottom],
        #      'Bottom_neutral': [neutral_list_bottom]}
        # df_LR_types = pd.DataFrame(data=d, columns=['Top_yes', 'Top_no', 'Top_neutral', 'Bottom_yes', 'Bottom_no',
        #                                             'Bottom_neutral'])
        # df_LR_types = df_LR_types.set_index('Top_yes')
        #
        # # Save LR results in a dataframe
        # result = pd.concat(frames)
        # result['Sample_name'] = values
        # result.set_index('Sample_name', inplace=True)
        #
        # # Save the results LR dataframe in a csv file.
        # try:
        #     with open(self.save_filename + '.csv', 'w') as f:
        #         result.to_csv(f)
        #     with open(self.save_filename + '.csv', 'a') as f:
        #         df_LR_types.to_csv(f)
        # except IOError:
        #     sys.exit()

        # button_load = Button(master, command=self.restart_program, text="Restart", height=2, width=15)
        # button_load.pack(side=TOP)

        mainloop()

    def load_data(self):
        # Load trained model
        if self.is_penile.get():
            filename = 'mlpmodel_penile'
        else:
            filename = 'mlpmodel'
        # If there is no input file selected, the program quits.
        try:
            # xl = pd.ExcelFile(self.open_filename)
            X_single, _, _, n_celltypes_with_penile, n_features, n_per_celltype, string2index, index2string,\
                marker_names, names = \
                get_data_per_cell_type(filename=self.open_filename, single_cell_types=self.single_cell_types,
                                   ground_truth_known=False, binarize=True,
                                       number_of_replicates = self.number_of_replicates)
        except ValueError:
            sys.exit()
        except IndexError:
            sys.exit()
        return filename, marker_names, names, X_single, n_celltypes_with_penile, n_features, n_per_celltype, string2index, index2string

    def add_buttons(self):
        for i, cell in enumerate(self.single_cell_types):
            self.top_variables[cell] = StringVar()
            t = Label(self.master, text=cell)
            t.grid(row=2, column=i + 4)
            rb = Radiobutton(self.master, text='Uniform', variable=self.top_variables[cell], value='Uniform')
            rb.grid(row=3, column=i + 4)
            rb.select()
            rb = Radiobutton(self.master, text='Always', variable=self.top_variables[cell], value='Always')
            rb.grid(row=4, column=i + 4)
            rb = Radiobutton(self.master, text='Never', variable=self.top_variables[cell], value='Never')
            rb.grid(row=5, column=i + 4)

            rb = Radiobutton(self.master, text='Uniform', variable=self.top_variables[cell], value='Uniform')
            rb.grid(row=7, column=i + 4)
            rb.select()
            rb = Radiobutton(self.master, text='Always', variable=self.top_variables[cell], value='Always')
            rb.grid(row=8, column=i + 4)
            rb = Radiobutton(self.master, text='Never', variable=self.top_variables[cell], value='Never')
            rb.grid(row=9, column=i + 4)

        self.open_text_widget = Text(self.master, height=2, width=50)
        self.open_text_widget.grid(row=2, column=3)
        self.open_text_widget.insert(END, 'Please select a file to load')
        button_load = Button(self.master, command=self.open_file_name, text="Load excel file", height=2, width=20)
        button_load.grid(row=2, column=2)

        self.save_text_widget = Text(self.master, height=2, width=50)
        self.save_text_widget.grid(row=3, column=3)
        self.save_text_widget.insert(END, 'Please select a location to save the results')
        button_load = Button(self.master, command=self.save_file_location, text="Set location to save results",
                             height=2, width=20)
        button_load.grid(row=3, column=2)

        button_load = Button(self.master, command=self.set_replicate_number, text="Set number of replicates", height=2,
                             width=20)
        button_load.grid(row=4, column=2)
        button = Checkbutton(
            self.master, text="Penile swab?", variable=self.is_penile,
        )
        button.grid(row=5, column=3)
        button_load = Button(self.master, command=self.close, text="Run", height=2, width=20, bg='darkseagreen')
        button_load.grid(row=9, column=2)


if __name__ == '__main__':
    # Create a window with buttons for selecting the input file, selecting the save location and a
    # button to run the calculations.

    app = AnalyseExcel()

    app.run()
