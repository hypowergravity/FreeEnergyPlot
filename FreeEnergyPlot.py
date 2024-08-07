# -*- coding: utf-8 -*-
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
import subprocess
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
import matplotlib
import matplotlib.ticker as mtick
matplotlib.use('Agg')
font = {'weight': 'bold', 'family': 'sans-serif', 'sans-serif': ['Helvetica']}

matplotlib.rc('font', **font)


class FreeEnergyPlot:
    """This script generates a plot of the per-residue energy decomposition calculated by AMBER's MMPBSA.py tool using MMPBSA_API. MMPBSA.py is a module in the AMBER molecular dynamics simulation software suite used to estimate the free energy of protein-ligand binding. The per-residue energy decomposition breaks down the total binding energy into contributions from each amino acid residue in the protein. The resulting plot provides insight into the key residues involved in the binding interaction."""

    def __init__(self, residues, residue_list=None, shift=0, skip_residue=None):
        self.data = MMPBSA_API.load_mmpbsa_info("_MMPBSA_info")
        self.df_complex = pd.DataFrame(self.data["gb"]["complex"]) if "pb" not in list(
            self.data.keys()) else pd.DataFrame(self.data["pb"]["complex"])
        self.df_receptor = pd.DataFrame(self.data["gb"]["receptor"]) if "pb" not in list(
            self.data.keys()) else pd.DataFrame(self.data["pb"]["receptor"])
        self.df_ligand = pd.DataFrame(self.data["gb"]["ligand"]) if "pb" not in list(
            self.data.keys()) else pd.DataFrame(self.data["pb"]["ligand"])
        self.df_delta = self.df_complex - self.df_receptor - self.df_ligand
        self.frames = [x + 1 for x in range(len(self.df_delta.index))]
        self.residues = residues
        self.residue_list = residue_list
        self.shift = shift
        self.skip_residue = skip_residue + self.shift if skip_residue is not None else None
        T_term, T_s_term, C_term, C_S_term = self.Entropy_term()

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
        self.df_total_decomposed_filtered = self.df_total_decomposed_filtered.fillna(
            value=0)
        self.df_total_decomposed_filtered.index = self.df_total_decomposed_filtered.index + self.shift
        self.df_total_decomposed_filtered = self.df_total_decomposed_filtered.drop(
            [self.skip_residue], axis=0) if self.skip_residue is not None else self.df_total_decomposed_filtered

        if "-" in str(self.df_total_decomposed_filtered.index[0]):
            self.decomposed_complex_update = self.Multi_index_set(
                self.decomposed_complex)
            self.decomposed_receptor_update = self.Multi_index_set(
                self.decomposed_receptor)
            self.ligand_residues = list(set(list(self.decomposed_complex_update.index.get_level_values(
                0))).difference(set(list(self.decomposed_receptor_update.index.get_level_values(0)))))

            if len(self.residues) > 0:
                for x in self.residues:
                    self.residue_wise_plot(x)

            for x in self.ligand_residues:
                self.residue_wise_plot(x)

            self.delta_decomposed_update = self.Multi_index_set(
                self.delta_decomposed)
            self.df_total_decomposed_filtered = self.delta_decomposed_update[(
                self.delta_decomposed_update >= 0.5) | (self.delta_decomposed_update <= -0.5)].dropna(how='all')
            self.df_total_decomposed_filtered = self.df_total_decomposed_filtered.fillna(
                value=0)
            self.df_total_decomposed_filtered["mean"] = self.df_total_decomposed_filtered.mean(
                axis=1)

            Residue_1_index = self.df_total_decomposed_filtered.index.get_level_values(
                0).astype(int).to_list()
            Residue_2_index = self.df_total_decomposed_filtered.index.get_level_values(
                1).astype(int).to_list()

            residue_list = list(
                set(Residue_1_index).union(set(Residue_2_index)))
            list.sort(residue_list)

            matrix = np.zeros((len(residue_list), len(residue_list)))
            for i, x in enumerate(residue_list):
                for j, y in enumerate(residue_list):
                    matrix[i, j] = self.data_value(
                        self.df_total_decomposed_filtered, x, y)

            self.pair_wise_plot = pd.DataFrame(
                matrix, index=residue_list, columns=residue_list)
            self.heatmap(self.pair_wise_plot, True, name="pair-wise")

        else:
            self.heatmap(self.df_total_decomposed_filtered, True, "0")
            self.heatmap(self.df_total_decomposed_filtered, False, "1")
            self.per_residue_bar_plot(self.df_total_decomposed_filtered)

    def Entropy_term(self):
        """ Estimation of the TΔS term as suggested by the paper
        On the Use of Interaction Entropy and Related Methods to Estimate Binding Entropies
        by Vilhelm Ekberg (PMC8389774)"""
        R = 0.001987
        temperature = 303.15
        deltaE_IE = np.array(self.df_delta["EEL"] + self.df_delta["VDWAALS"])
        const_factor = R * temperature

        internal_energy_term = (deltaE_IE - deltaE_IE.mean())/const_factor
        T_delta_S = const_factor * \
            np.log(np.array(np.exp(internal_energy_term)).mean())
        T_delta_S_individual = const_factor * \
            np.log(np.array(np.exp(internal_energy_term)))
        str_1 = "T\u0394S entropy term (Interaction Entropy)"
        str_2 = "and its standard deviation\n estimated by the method suggested by Zhang and co-workers 2016;"
        str_3 = "T\u0394S: %.3f, std: +/- %.3f" % (
            T_delta_S, T_delta_S_individual.std())
        template_file = str_1 + str_2 + str_3
        print(template_file)
        with open("entropy.txt", "w") as file:
            file.write(template_file)

        size = deltaE_IE.size
        array_of_c2 = np.zeros(2000)
        for i in range(2000):
            idxs = np.random.randint(0, size, size)
            ie_std = deltaE_IE[idxs].std()
            c2data = (ie_std ** 2) / (2 * temperature * R)
            array_of_c2[i] = c2data
        c2_std = float(np.sort(array_of_c2)[100:1900].std())
        str_4 = "T\u0394S entropy term (C2 - Interaction Entropy) and its standard deviation \n"
        str_5 = "estimated by method suggested by  Minh and co-workers 2018 ;"
        str_6 = "T\u0394S: %.3f, std: +/- %.3f" % (c2data, c2_std)
        template_file_2 = str_4 + str_5 + str_6
        print(template_file_2)
        with open("entropy.txt", "a") as file:
            file.write(template_file_2)
        return T_delta_S, T_delta_S_individual.std(), c2data, c2_std

    def residue_wise_plot(self, residue):
        df_lig = self.decomposed_complex_update[self.decomposed_complex_update.index.get_level_values(
            0) == residue]
        df_filtered_lig = df_lig[(df_lig >= 0.5) | (
            df_lig <= -0.5)].dropna(how='all')
        df_filtered_lig = df_filtered_lig.fillna(value=0)
        df_filtered_heat_plt = df_filtered_lig.set_index(
            df_filtered_lig.index.get_level_values(1))
        df_filtered_lig_plot = pd.DataFrame({"Index": list(
            df_filtered_heat_plt.index), "Values": list(df_filtered_heat_plt.mean(axis=1))})
        df_filtered_lig_plot = df_filtered_lig_plot.set_index("Index")
        df_filtered_lig_plot = df_filtered_lig_plot[df_filtered_lig_plot.index != residue]
        self.heatmap(df_filtered_heat_plt, True, name="_%s" % residue)
        self.heatmap(df_filtered_heat_plt, False, name="_f_%s" % residue)
        self.per_residue_bar_plot(df_filtered_lig_plot, name="_%s" % residue)

    @staticmethod
    def data_value(df, i, j):
        try:
            return df.loc[(df.index.get_level_values(0).astype(int) == i) & (df.index.get_level_values(1).astype(int) == j), 'mean'].iloc[0]

        except IndexError:
            return 0

    def Multi_index_set(self, df):
        df = df.set_index(pd.MultiIndex.from_tuples(
            df.index.str.split("-", expand=True)))
        df.index = df.index.set_levels(df.index.levels[0].astype(int), level=0)
        df.index = df.index.set_levels(df.index.levels[1].astype(int), level=1)
        return df

    def plot_line(self):
        avg_y = np.mean(self.df_delta["TOTAL"])
        fig, ax = plt.subplots()

        ax.plot([x for x in range(len(self.df_delta.index))],
                self.df_delta["TOTAL"], label='Total')
        ax.axhline(y=avg_y, color='red', linestyle='--', label='Average')

        ax.set_xticks(self.df_delta.index)
        ax.set_xticklabels([x+1 for x in self.df_delta.index], rotation=90)
        ax.set_xlim(min(self.df_delta.index), max(self.df_delta.index))
        if len(self.frames) > 10:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='lower'))
            tick_formatter = ScalarFormatter(useOffset=True, useMathText=True)
            tick_formatter.set_powerlimits(
                (int(min(self.df_delta.index)), int(max(self.df_delta.index))))
            ax.xaxis.set_major_formatter(tick_formatter)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(
                nbins=len(self.frames), prune='lower'))

        plt.title(r'Energy component $\Delta$G Total as a function of frames')
        plt.xlabel('Frames')
        plt.ylabel('Energy (kcal/mol)')
        plt.legend()
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

        ax.errorbar(data_frame.columns, means, yerr=errors,
                    fmt='none', capsize=3, ecolor='black')
        plt.title('Energy components with standard errors')
        plt.xlabel('Energy (kcal/mol)')
        plt.ylabel('Energy components')
        plt.xticks(data_frame.columns, list(data_frame.columns), rotation=90)
        plt.tight_layout()
        plt.savefig("delta_energy_bar_plot_%s.png" % name, dpi=600)
        plt.close()

    def energy_decomposition(self, name):
        decomp_nested_dict = self.data["decomp"]["gb"][name]["TDC"] if "pb" not in list(
            self.data.keys()) else self.data["decomp"]["pb"][name]["TDC"]
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

        df_decomp_total.columns = ["frame_%s" %
                                   x for x in range(len(df_decomp_total.columns))]

        return df_decomp_total

    def heatmap(self, df, annot=True, name=""):
        fig, ax = plt.subplots()
        ax = sns.heatmap(df, cmap='Spectral', annot=annot,
                         fmt='.1f', linewidths=.5, xticklabels=1, yticklabels=1)

        if self.residue_list is not None:
            y_tick_labels = [int(tick) for tick in list(df.index)]
            y_tick_labels_1 = list(self.residue_list[self.residue_list['Resn'].isin(
                y_tick_labels)].apply(lambda x: '{}-{}'.format(x[0], x[1]), axis=1))
        else:
            y_tick_labels_1 = [int(tick) for tick in list(df.index)]

        ax.yaxis.set_major_locator(MaxNLocator())
        ax.set_yticks(np.arange(len(y_tick_labels_1)) + 0.5)
        ax.set_yticklabels(y_tick_labels_1, ha="center", minor=False)
        plt.setp(ax.get_yticklabels(), rotation=0,
                 ha="center", rotation_mode="anchor")
        ax.tick_params(axis='y', which='major', pad=20)

        X_ticks = [x+1 for x in range(len(df.columns))]
        plt.xticks(np.arange(len(X_ticks))+0.5, X_ticks, rotation=90)
        if len(self.frames) > 10:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='lower'))
            tick_formatter = ScalarFormatter(useOffset=True, useMathText=True)
            tick_formatter.set_powerlimits((min(X_ticks), max(X_ticks)))
            ax.xaxis.set_major_formatter(tick_formatter)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(
                nbins=max(X_ticks), prune='lower'))
            tick_formatter = ScalarFormatter(useOffset=True, useMathText=True)
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        plt.title('Per-residue energy decomposition plot')
        plt.xlabel('Frames')
        plt.ylabel('Residues')
        plt.tight_layout()
        plt.savefig("delta_energy_per-residue_heatmap_%s.png" % name, dpi=600)
        plt.close()

    def per_residue_bar_plot(self, df, name=""):
        fig, ax = plt.subplots()
        ax = sns.barplot(data=df.T, capsize=0.1)
        plt.title('Energy components Per Residue Decomposition')
        plt.xlabel('Residue')
        plt.ylabel('Energy (kcal/mol)')
        x_ticks = plt.xticks()[0]
        if self.residue_list is not None:
            x_tick_labels = [int(tick) for tick in df.index]
            x_tick_labels_1 = list(self.residue_list[self.residue_list['Resn'].isin(
                x_tick_labels)].apply(lambda x: '{}-{}'.format(x[0], x[1]), axis=1))
            ax.tick_params(axis='x', which='major', pad=25)
        else:
            x_tick_labels_1 = [int(tick) for tick in df.index]
            ax.tick_params(axis='x', which='major', pad=5)
        ax.set_xticklabels(x_tick_labels_1, ha="center",
                           rotation_mode='anchor')
        plt.xticks(x_ticks, x_tick_labels_1, rotation=90, ha='center')

        # ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        plt.tight_layout()
        plt.savefig("delta_energy_per-residue_barplot%s.png" % name, dpi=600)
        plt.close()


def main():
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(prog='FreeEnergyPlot.py', formatter_class=argparse.RawTextHelpFormatter,
                                     description="""This script analyse and plots per-residue/ pair-wise free energy decomposition using MMGBSA_API,
                                      \n for pair-wise decomposition, a list of residues can be given. 
                                      \n previous calculation of MMPBSA/MMGBSA has to be done previously, for details please refer Readme.MD file. 
                                      \n The script assumes the MMPB/GBSA is successful and the folder contains "_MMPBSA_info" file for ploting. 
                                      \n if residue -r has to be given if the the residue name should also be plotted.
        Few example two run the code: \n\n

        1. python FreeEnergyPlot.py \n
        2. python FreeEnergyPlot.py -n 25 26 169 215 272 300 \n
        3. python FreeEnergyPlot.py -r ":1-304" -p sanga1_protein.prmtop -n 25 26 169 215 272 --skip 69 \n
    
        

    """)

    parser.add_argument('-r', "--residuelist", action='store', type=str,
                        help='the list of residue that are in the pdb as provided in prmtop file')
    parser.add_argument('-p', "--prmtop", action='store',
                        type=str, help='parmtop file for getting the label')
    parser.add_argument('-n', "--residue", metavar='N', action='store', type=int, nargs='+',
                        help='a list of residues for \n for pair-wise decomposition by default lingand is choosen')
    parser.add_argument('-s', "--shift", action='store',
                        type=int, help='shift in number')
    parser.add_argument('--skip', action='store',
                        type=int, help='skip the residue')

    args = parser.parse_args()

    residues = [] if args.residue is None else args.residue
    residuelist = ":1-1000" if args.residuelist is None else args.residuelist
    prmtop = None if args.prmtop is None else args.prmtop
    shift = 0 if args.shift is None else args.shift
    skip_residue = None if args.skip is None else int(args.skip)
    start = datetime.now()
    if prmtop is not None and prmtop.endswith(".prmtop"):
        template = "cpptraj -p {prmtop} --resmask {residuelist} | awk 'BEGIN {{OFS=\",\"}} {{print $2,$1}}' > resilist.txt".format(
            prmtop=prmtop, residuelist=residuelist)
        result = subprocess.run([template], shell=True)
        residue_list = pd.read_csv("resilist.txt", names=[
                                   "Resi", "Resn"], skiprows=1)
        residue_list["Resn"] = residue_list["Resn"].astype(int) + shift
        FreeEnergyPlot(residues, residue_list, shift=shift,
                       skip_residue=skip_residue)
    else:
        FreeEnergyPlot(residues, shift=shift, skip_residue=skip_residue)
    print(datetime.now() - start)


if __name__ == '__main__':
    main()
