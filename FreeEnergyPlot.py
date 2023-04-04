import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from MMPBSA_mods import API as MMPBSA_API


class FreeEnergyPlot:
    """This script generates a plot of the per-residue energy decomposition calculated by AMBER's MMPBSA.py tool using MMPBSA_API. MMPBSA.py is a module in the AMBER molecular dynamics simulation software suite used to estimate the free energy of protein-ligand binding. The per-residue energy decomposition breaks down the total binding energy into contributions from each amino acid residue in the protein. The resulting plot provides insight into the key residues involved in the binding interaction."""
    def __init__(self):
        self.data = MMPBSA_API.load_mmpbsa_info("_MMPBSA_info")
        self.df_complex = pd.DataFrame(self.data["gb"]["complex"])
        self.df_receptor = pd.DataFrame(self.data["gb"]["receptor"])
        self.df_ligand = pd.DataFrame(self.data["gb"]["ligand"])
        self.df_delta = self.df_complex - self.df_receptor - self.df_ligand
        self.frames = [x + 1 for x in range(len(self.df_delta.index))]

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
        self.heatmap(True,"0")
        self.heatmap(False,"1")
        self.per_residue_bar_plot()

    def plot_line(self):
        avg_y = np.mean(self.df_delta["TOTAL"])
        plt.plot([x for x in range(len(self.df_delta.index))], self.df_delta["TOTAL"], label='Total')
        plt.axhline(y=avg_y, color='red', linestyle='--', label='Average')
        plt.xticks(self.df_delta.index, self.frames, rotation=90)
        plt.title(r'Energy component \Delta Total as a function of frames')
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

        ax.errorbar(data_frame.columns, means, yerr=errors, fmt='none', capsize=3, ecolor='black')
        plt.title('Energy components with standard errors')
        plt.xlabel('Energy (kcal/mol)')
        plt.ylabel('Energy components')
        plt.xticks(data_frame.columns, list(data_frame.columns), rotation=90)
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

    def heatmap(self,annot=True,name=""):
        fig, ax = plt.subplots()
        ax = sns.heatmap(self.df_total_decomposed_filtered, cmap='Spectral', annot=annot, fmt='.1f', linewidths=.5)
        ax.set_xticks(self.frames)
        ax.set_xticklabels(self.frames, rotation=90, ha="center", rotation_mode='anchor')
        plt.title('Per-residue energy decomposition plot')
        plt.xlabel('Frames')
        plt.ylabel('Residues')
        plt.tight_layout()
        plt.savefig(f"delta_energy_per-residue_heatmap_%s.png"%name, dpi=600)
        plt.close()

    def per_residue_bar_plot(self):
        fig, ax = plt.subplots()
        ax = sns.barplot(data=self.df_total_decomposed_filtered.T, ci='sd', capsize=0.1)
        plt.title('Energy components Per Residue Decomposition')
        plt.xlabel('Residue')
        plt.ylabel('Energy (kcal/mol)')
        plt.xticks(rotation=90, ha='center')
        plt.tight_layout()
        plt.savefig(f"delta_energy_per-residue_barplot.png", dpi=600)
        plt.close()


def run():
    FreeEnergyPlot()


if __name__ == "__main__":
    run()
