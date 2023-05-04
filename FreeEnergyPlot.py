"""
Script Name: FreeEnergyPlot.py
Author: Sriram Srinivasa Raghavan (hypowergravity@gmail.com)
Description: The script plot per residue and pair wise energy decomposition using MMPBSA_API 
License: MIT License (https://opensource.org/licenses/MIT)
"""



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from MMPBSA_mods import API as MMPBSA_API
from matplotlib.ticker import MaxNLocator


class FreeEnergyPlot:
    """This script generates a plot of the per-residue energy decomposition calculated by AMBER's MMPBSA.py tool using MMPBSA_API. MMPBSA.py is a module in the AMBER molecular dynamics simulation software suite used to estimate the free energy of protein-ligand binding. The per-residue energy decomposition breaks down the total binding energy into contributions from each amino acid residue in the protein. The resulting plot provides insight into the key residues involved in the binding interaction."""
    def __init__(self,residues):
        self.data = MMPBSA_API.load_mmpbsa_info("_MMPBSA_info")
        self.df_complex = pd.DataFrame(self.data["gb"]["complex"])
        self.df_receptor = pd.DataFrame(self.data["gb"]["receptor"])
        self.df_ligand = pd.DataFrame(self.data["gb"]["ligand"])
        self.df_delta = self.df_complex - self.df_receptor - self.df_ligand
        self.frames = [x + 1 for x in range(len(self.df_delta.index))]
        self.residues = residues
        

        self.plot_line()

        bar_plot = {"delta": self.df_delta, "complex": self.df_complex, "receptor": self.df_receptor,
                    "ligand": self.df_ligand}
        [self.bar_plot(v, k) for k, v in bar_plot.items()]
        self.decomposed_complex = self.energy_decomposition("complex")
        self.decomposed_receptor = self.energy_decomposition("receptor")

        

        self.delta_decomposed = self.decomposed_complex - self.decomposed_receptor

        self.df_total_decomposed_filtered = self.delta_decomposed[
            (self.delta_decomposed >= 0.5) | (
                    self.delta_decomposed <= -0.5)].dropna(how='all')
        self.df_total_decomposed_filtered = self.df_total_decomposed_filtered.fillna(value=0)

        if "-" in str(self.df_total_decomposed_filtered.index[0]):
            self.decomposed_complex_update = self.Multi_index_set(self.decomposed_complex)
            self.decomposed_receptor_update = self.Multi_index_set(self.decomposed_receptor)
            self.ligand_residues = list(set(list(self.decomposed_complex_update.index.get_level_values(0))).difference(set(list(self.decomposed_receptor_update.index.get_level_values(0)))))
            
            if len(self.residues) > 0:
                for x in self.residues:
                    self.residue_wise_plot(x)

            for x in self.ligand_residues:
                self.residue_wise_plot(x)

            self.delta_decomposed_update = self.Multi_index_set(self.delta_decomposed)
            self.df_total_decomposed_filtered = self.delta_decomposed_update[(self.delta_decomposed_update >= 0.5) | (self.delta_decomposed_update <= -0.5)].dropna(how='all')
            self.df_total_decomposed_filtered = self.df_total_decomposed_filtered.fillna(value=0)
            self.df_total_decomposed_filtered["mean"] = self.df_total_decomposed_filtered.mean(axis=1)

            Residue_1_index = self.df_total_decomposed_filtered.index.get_level_values(0).astype(int).to_list()
            Residue_2_index = self.df_total_decomposed_filtered.index.get_level_values(1).astype(int).to_list()

            residue_list = list(set(Residue_1_index).union(set(Residue_2_index)))
            list.sort(residue_list)
            
            matrix = np.zeros((len(residue_list), len(residue_list)))
            for i,x in enumerate(residue_list):
                for j,y in enumerate(residue_list):
                        matrix[i, j] = self.data_value(self.df_total_decomposed_filtered,x,y)
            
            self.pair_wise_plot = pd.DataFrame(matrix, index =residue_list, columns = residue_list)
            self.heatmap_pair_wise(self.pair_wise_plot,True,name="pair-wise")
            self.heatmap_pair_wise(self.pair_wise_plot,False,name="pair-wise_1")

        else:
            self.heatmap(self.df_total_decomposed_filtered,True,"0")
            self.heatmap(self.df_total_decomposed_filtered,False,"1")
            self.per_residue_bar_plot(self.df_total_decomposed_filtered)


    def residue_wise_plot(self,residue):
        df_lig = self.decomposed_complex_update[self.decomposed_complex_update.index.get_level_values(0)==residue]
        df_filtered_lig = df_lig[(df_lig >= 0.5) | (df_lig <= -0.5)].dropna(how='all')
        df_filtered_lig = df_filtered_lig.fillna(value=0)
        df_filtered_heat_plt = df_filtered_lig.set_index(df_filtered_lig.index.get_level_values(1))
        df_filtered_lig_plot = pd.DataFrame({"Index":list(df_filtered_heat_plt.index),"Values":list(df_filtered_heat_plt.mean(axis=1))})
        df_filtered_lig_plot = df_filtered_lig_plot.set_index("Index")
        df_filtered_lig_plot = df_filtered_lig_plot[df_filtered_lig_plot.index !=residue]
        self.heatmap(df_filtered_heat_plt,True,name=f"_%s"%residue)
        self.heatmap(df_filtered_heat_plt,False,name=f"_f_%s"%residue)
        self.per_residue_bar_plot(df_filtered_lig_plot,name=f"_%s"%residue)
        
    @staticmethod
    def data_value(df, i,j):
        try:
            return df.loc[(df.index.get_level_values(0).astype(int) == i) & (df.index.get_level_values(1).astype(int) == j), 'mean'].iloc[0]

        except IndexError:
            return 0

    def Multi_index_set(self,df):
        df = df.set_index(pd.MultiIndex.from_tuples(df.index.str.split("-",expand=True)))
        df.index = df.index.set_levels(df.index.levels[0].astype(int),level=0)
        df.index = df.index.set_levels(df.index.levels[1].astype(int),level=1)
        return df


    def plot_line(self):
        avg_y = np.mean(self.df_delta["TOTAL"])
        fig, ax = plt.subplots()
        ax.plot([x for x in range(len(self.df_delta.index))], self.df_delta["TOTAL"], label='Total')
        plt.axhline(y=avg_y, color='red', linestyle='--', label='Average')
        plt.xticks(self.df_delta.index, self.frames, rotation=90)
        plt.title(r'Energy component $\Delta$G Total as a function of frames')
        plt.xlabel('Frames')
        plt.ylabel('Energy (kcal/mol)')
        plt.legend()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        plt.tight_layout()
        plt.savefig("delta_energy_wrt_time.png", dpi=600)
        plt.close()

    def bar_plot(self, data_frame, name):
        means = data_frame.mean()
        errors = data_frame.sem()
        # colors = ['blue', 'green', 'red', 'orange']
        fig, ax = plt.subplots()
        ax.bar(data_frame.columns, means, color="blue", alpha=0.5)

        # set x-axis label rotation
        ax.tick_params(axis='x', rotation=45)

        # set transparency and color for each individual bar
        for i, column in enumerate(data_frame.columns):
            ax.bar(column, data_frame[column].mean(), color='blue', alpha=0.8)

        ax.errorbar(data_frame.columns, means, yerr=errors, fmt='none', capsize=3, ecolor='black')
        plt.title('Energy components with standard errors')
        plt.xlabel('Energy (kcal/mol)')
        plt.ylabel('Energy components')
        plt.xticks(data_frame.columns, list(data_frame.columns), rotation=90)
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        plt.tight_layout()
        plt.savefig(f"delta_energy_bar_plot_%s.png" % name, dpi=600)
        plt.close()

    def energy_decomposition(self, name):
        decomp_nested_dict = self.data["decomp"]["gb"][name]["TDC"]
        grouped_dict = {residue: {} for residue in decomp_nested_dict}
        for residue, values_dict in decomp_nested_dict.items():
            for property_name, values_array in values_dict.items():
                for i in range(len(values_array)):
                    if i not in grouped_dict[residue]:
                        grouped_dict[residue][i] = {}
                    grouped_dict[residue][i][property_name] = values_array[i]

        # convert grouped dictionary to dataframe
        df_decomp = pd.DataFrame.from_dict(grouped_dict, orient='index')

        def data_frame(df, column, frame):
            return pd.DataFrame(df.to_dict()[frame]).T[column]

        df_decomp_total = pd.DataFrame()
        for x in range(len(df_decomp.columns)):
            df_decomp_total = pd.concat([df_decomp_total, data_frame(df_decomp, "tot", x)],
                                        axis=1)

        df_decomp_total.columns = [f"frame_%s" % x for x in range(len(df_decomp_total.columns))]

        return df_decomp_total

    def heatmap(self,df,annot=True,name=""):
        fig, ax = plt.subplots()
        ax = sns.heatmap(df, cmap='Spectral', annot=annot, fmt='.1f', linewidths=.5,xticklabels=1, yticklabels=1)
        ax.set_xticks(self.frames)
        ax.set_xticklabels(self.frames, rotation=90, ha="center", rotation_mode='anchor')
        # ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        # print(list(self.df_total_decomposed_filtered.index),len(list(self.df_total_decomposed_filtered.index)))
        # y_ticks = plt.yticks()[0]
        # print(y_ticks)
        y_tick_labels =[int(tick) for tick in list(df.index)]
        ax.yaxis.set_major_locator(MaxNLocator())
        #ax.set_yticks(list(self.df_total_decomposed_filtered.index))
        ax.set_yticks(np.arange(len(y_tick_labels))+ 0.5)
        ax.set_yticklabels(y_tick_labels, ha="center",minor=False)
        plt.setp(ax.get_yticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        ax.tick_params(axis='y', which='major', pad=12)
        ax.tick_params(axis='x', which='major', pad=12)
        #ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # print(y_tick_labels,len(y_tick_labels))
        plt.title('Per-residue energy decomposition plot')
        plt.xlabel('Frames')
        plt.ylabel('Residues')
        plt.tight_layout()
        plt.savefig(f"delta_energy_per-residue_heatmap_%s.png"%name, dpi=600)
        plt.close()

    def heatmap_pair_wise(self,df,annot=False,name=""):
        fig, ax = plt.subplots()
        # y_tick_labels =[int(tick) for tick in list(df.index)]
        ax = sns.heatmap(df, cmap='Spectral', annot=annot, fmt='.1f', linewidths=.5,xticklabels=1, yticklabels=1)
        # ax.set_xticks(y_tick_labels)
        # ax.set_xticklabels(y_tick_labels, rotation=90, ha="center", rotation_mode='anchor')
        # ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        # # print(list(self.df_total_decomposed_filtered.index),len(list(self.df_total_decomposed_filtered.index)))
        # # y_ticks = plt.yticks()[0]
        # # print(y_ticks)
        # # y_tick_labels =[int(tick) for tick in list(df.index)]
        # ax.yaxis.set_major_locator(MaxNLocator())
        # #ax.set_yticks(list(self.df_total_decomposed_filtered.index))
        # ax.set_yticks(np.arange(len(y_tick_labels))+ 0.5)
        # ax.set_yticklabels(y_tick_labels, ha="center",minor=False)
        # plt.setp(ax.get_yticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        # ax.tick_params(axis='y', which='major', pad=12)
        # ax.tick_params(axis='x', which='major', pad=12)
        #ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # print(y_tick_labels,len(y_tick_labels))
        plt.title('Per-residue energy decomposition plot')
        plt.xlabel('Residues')
        plt.ylabel('Residues')
        plt.tight_layout()
        plt.savefig(f"delta_energy_residue_heatmap_%s.png"%name, dpi=600)
        plt.close()

    def per_residue_bar_plot(self,df,name=""):
        fig, ax = plt.subplots()
        ax = sns.barplot(data=df.T, capsize=0.1)
        plt.title('Energy components Per Residue Decomposition')
        plt.xlabel('Residue')
        plt.ylabel('Energy (kcal/mol)')
        x_ticks = plt.xticks()[0]
        x_tick_labels =[f"{int(tick)}" for tick in df.index]
        ax.set_xticklabels(x_tick_labels, ha="center", rotation_mode='anchor')
        plt.xticks(x_ticks,x_tick_labels,rotation=90, ha='center')
        ax.tick_params(axis='x', which='major', pad=12)

        #ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        plt.tight_layout()
        plt.savefig(f"delta_energy_per-residue_barplot%s.png"%name, dpi=600)
        plt.close()



def main():
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(prog='FreeEnergyPlot.py', formatter_class=argparse.RawTextHelpFormatter,
                                     description=f"""This script analyse and plots per-residue/ pair-wise free energy decomposition using MMGBSA_API,
                                      \n for pair-wise decomposition, a list of residues can be given. 
                                      \n previous calculation of MMPBSA/MMGBSA has to be done previously, for details please refer Readme.MD file. 
                                      \n The script assumes the MMPB/GBSA is successful and the folder contains "_MMPBSA_info" file for ploting. 
        Few example two run the code: \n\n
        1. python FreeEnergyPlot.py  \n\n
        2. python FreeEnergyPlot.py -n 25 26 169 215 272 300

    """)
    parser.add_argument('-n',"--residue", metavar='N',action='store', type=int, nargs='+', help='a list of residues for \n for pair-wise decomposition by default lingand is choosen')

    args = parser.parse_args()

    residues = [] if args.residue is None else args.residue            
    start = datetime.now()
    FreeEnergyPlot(residues)  
    print(datetime.now() - start)         

if __name__ == '__main__':
    main()

