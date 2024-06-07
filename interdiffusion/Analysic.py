import os, ast
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import multiprocessing as mp
from matplotlib.ticker import ScalarFormatter
from functools import partial


class MonoInterdiffusionAnalyzer:
    def __init__(
        self,
        element,
        concentration,
        recordsteps,
        alloy_system,
        lattice_record,
        v_position_record,
        lattice_size,
        plane_length,
        couple_length,
        vacancy_record,
        KMCsteps,

    ):
        self.alloy_system = alloy_system
        self.lattice_size = lattice_size
        self.plane_length = plane_length * self.lattice_size
        self.couple_length = couple_length * lattice_size
        self.element = element
        self.concentration = concentration
        self.plane_atom_num = self.plane_length**2
        self.recordsteps = recordsteps
        self.successtrial = len(lattice_record)
        self.element_type = ["A", "B", "C", "D", "E"]
        self.lattice_record = lattice_record
        self.v_position_record = v_position_record
        self.maindir = os.path.abspath(os.curdir)
        self.spacing = 2

    def extract_step_lattice_record(self, step):
        return {
            walker: np.array(step_record[step])
            for walker, step_record in self.lattice_record.items()
            if step in step_record
        }

    def position_composition(self, lattice):
        """
        根據傳入的Lattice array紀錄, 回傳不同Plane上的不同元素的佔比, 為字典形式
        """
        composition_record = {}
        for i, x in enumerate(
            range(0, 2 * self.couple_length * self.plane_atom_num, self.plane_atom_num)
        ):
            atoms = lattice[x : x + 1 * self.plane_atom_num]
            element, concentration = np.unique(atoms, return_counts=True)
            composition_record[i] = dict(zip(element, concentration / len(atoms)))
        return composition_record

    def concentration_profile(self, step):
        # 分析不同步數下元素在各面上的比例
        record = self.extract_step_lattice_record(step)
        concentration_df = pd.DataFrame()
        dfs_to_concat = []
        for walker in record:
            composition = self.position_composition(record[walker])
            composition_df = pd.DataFrame.from_dict(
                composition, orient="index"
            ).sort_index()
            walker_df = pd.DataFrame(index=composition_df.index)
            for column in composition_df.columns:
                if column != 0:
                    walker_df[f"{walker}_{chr(64+column)}"] = composition_df[column]
            dfs_to_concat.append(walker_df)
        concentration_df = pd.concat(dfs_to_concat, axis=1).fillna(0)
        result_df = pd.DataFrame(index=composition_df.index)
        # 將多次結果平均
        for element in self.element_type:
            columns_with_element = concentration_df.columns[
                concentration_df.columns.str.contains(f"_{element}")
            ]
            if columns_with_element.any():
                result_df[element] = concentration_df[columns_with_element].mean(axis=1)
                result_df[f"{element}_error"] = concentration_df[
                    columns_with_element
                ].std(axis=1)

        # 取得濃度對位置一次與二次微分
        result_df = self.concentration_derivative(result_df)
        result_df.to_excel(f"Concentration as position of {step}.xlsx")
        return result_df

    def concentration_derivative(self, concentration_df):
        for element in self.element_type:
            if element in concentration_df.columns:
                # First derivative of concentration
                concentration_df[f"Flux {element}"] = (
                    -concentration_df[f"{element}"].diff(periods=-1)
                    + concentration_df[f"{element}"].diff()
                ) * 0.5
                concentration_df.fillna(0)
                # Second derivative of concentration
                concentration_df[f"Accumulation {element}"] = (
                    -concentration_df[f"Flux {element}"].diff(periods=-1)
                    + concentration_df[f"Flux {element}"].diff()
                ) * 0.5
                concentration_df.fillna(0)

        return concentration_df

    def draw_concentration_profile(self, step, concentration_df):
        for element in self.element_type:
            if element in concentration_df.columns:
                plt.plot(
                    concentration_df.index,
                    concentration_df[element],
                    marker="s",
                    linestyle="-",
                    label=element,
                )
        plt.legend()
        plt.title(f"After {step} steps.")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"Final After{step} steps.jpg")
        plt.close()

    def draw_enlarge_concentration_profile(self, step, concentration_df):
        fig, ax = plt.subplots()
        # Original Plot
        elementlist = []
        for element in self.element_type:
            if element in concentration_df.columns:
                elementlist.append(element)
                ax.plot(
                    concentration_df.index,
                    concentration_df[element],
                    marker="s",
                    linestyle="-",
                    label=element,
                )
        ax.legend()
        ax.set_title(f"After {step} steps.")
        ax.set_xlabel("Position")
        ax.set_ylabel("Concentration")

        x1, x2, y1, y2 = (
            self.couple_length - 5,
            self.couple_length + 5,
            min(concentration_df[elementlist].min()) - 0.1,
            max(concentration_df[elementlist].max()) + 0.1,
        )  # subregion of the original image
        axins = ax.inset_axes(
            [0.55, 0.55, 0.3, 0.3],
            xlim=(x1, x2),
            ylim=(y1, y2),
            xticklabels=[],
            yticklabels=[],
        )
        # Enlarged Plot
        for element in self.element_type:
            if element in concentration_df.columns:
                axins.errorbar(
                    concentration_df.index,
                    concentration_df[element],
                    yerr=concentration_df[f"{element}_error"],
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    label=element,
                )
        ax.indicate_inset_zoom(axins, edgecolor="black")
        # Adjust the layout for the entire figure
        plt.tight_layout()
        plt.title(f"After {step} steps")
        # Show the figure
        plt.savefig(f"Enlarge inside After {step} steps.jpg")
        # plt.show()
        plt.close()

    def draw_flux_profile(self, step, concentration_df):
        fig, ax = plt.subplots(ncols=1, nrows=2)
        plt.rc("text", usetex=True)
        for element in self.element_type:
            if element in concentration_df.columns:
                # 子圖1: Flux
                ax[0].plot(
                    concentration_df.index,
                    concentration_df[f"Flux {element}"],
                    marker="o",
                    linestyle="-",
                    label=f"{element}",
                )

                # 子圖2: Accumulation
                ax[1].plot(
                    concentration_df.index,
                    concentration_df[f"Accumulation {element}"],
                    marker="o",
                    linestyle="-.",
                    label=f"{element}",
                )

        ax[0].set_title("Flux")
        ax[0].set_xlabel("Position")
        ax[0].set_ylabel(r"$\partial C / \partial x$")
        ax[0].legend()
        ax[0].grid(True)

        ax[1].set_title("Accumulation")
        ax[1].set_xlabel("Position")
        ax[1].set_ylabel(r"$\partial^2 C / \partial x^2$")
        ax[1].legend()
        ax[1].grid(True)

        plt.suptitle("Flux and Accumulation Profiles")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"Derivative of concentration profiles of {step}.jpg")
        plt.close()

    def concentration_analysis(self):
        os.chdir("simulationresult")
        if not os.path.exists(r"concentration"):
            os.makedirs(r"concentration")
        os.chdir("concentration")
        for step in self.recordsteps:
            # Convert lattice information into concetration as function of position
            concentration_df = self.concentration_profile(step)
            # Plot whole concentration profile without/ with enlarge plot
            self.draw_concentration_profile(step, concentration_df)
            self.draw_enlarge_concentration_profile(step, concentration_df)
            # Plot derivative of concentration profile, corresponding to flux and accumulation
            self.draw_flux_profile(step, concentration_df)
        os.chdir(self.maindir)
        return

    def step_vacancy_backhop_probability(self):
        if not os.path.exists("backhopresult"):
            os.makedirs("backhopresult")
        os.chdir("backhopresult")
        for step in self.record_step[1:]:
            backhop_probability_record = np.empty(self.trial_times)
            for trial, (x, y, z) in enumerate(
                zip(self.x_columns, self.y_columns, self.z_columns)
            ):
                df = pd.DataFrame()
                difference = (
                    self.vacancy_record[[x, y, z]]
                    .iloc[:step]
                    .diff()
                    .fillna(value=self.vacancy_record.iloc[[0]])
                )
                shift = difference.shift(periods=1).fillna(0)
                df[x] = difference[x] + shift[x]
                df[y] = difference[y] + shift[y]
                df[z] = difference[z] + shift[z]
                backhop_probability_record[trial] = (
                    df[(df[x] == 0) & (df[y] == 0) & (df[z] == 0)].shape[0]
                    / self.vacancy_record.iloc[:step].shape[0]
                )
            mean_probability = np.mean(backhop_probability_record)
            std_probability = np.std(backhop_probability_record)
            with open(f"Backhop Probability for Vacancy at {step}.txt", mode="w") as fh:
                fh.write(
                    f"Backhop probability of {self.alloy_system} with {mean_probability} +/- {std_probability} \n"
                )
                for pro in backhop_probability_record:
                    fh.write(f"{str(pro)}\n")

            # 繪製直方圖
            plot_backhoppro_detail = {
                "title": "Back hop probability Distribution",
                "xlabel": "Probability",
                "ylabel": "Normalized Constant",
                "filename": f"Backhop probability distribution at {step} steps.png",
            }
            self.result_histo_plotting(
                backhop_probability_record, plot_backhoppro_detail
            )
        os.chdir(self.maindir)
        return
    
    def step_vacancy_site_sd_recording(self):
        if not os.path.exists("sitesdresult"):
            os.makedirs("sitesdresult")
        os.chdir("sitesdresult")
        for step in self.recordsteps:
            sites_visited_result = np.empty(self.trial_times)
            sd_result = np.empty(self.trial_times)
            for trial, (trial_x, trial_y, trial_z, sd) in enumerate(
                zip(self.x_columns, self.y_columns, self.z_columns, self.sd_columns)
            ):
                df = self.vacancy_record[[trial_x, trial_y, trial_z]].iloc[:step].copy()
                df["count"] = 1
                site_couts = (
                    df.groupby([trial_x, trial_y, trial_z])
                    .agg({"count": len})
                    .reset_index()
                )
                num_sites_visited = site_couts.shape[0]
                sites_visited_result[trial] = num_sites_visited
                sd_result[trial] = self.vacancy_record[sd].iloc[step - 1]
            sites_mean = np.mean(sites_visited_result)
            sites_median = np.median(sites_visited_result)
            sd_mean = np.mean(sd_result)
            sd_median = np.median(sd_result)
            with open(f"Site visited and MSD result at {step}.txt", mode="w") as fh:
                fh.write(
                    f"{self.alloy_system}, {(sites_mean,sites_median,sd_mean,sd_median)}\n"
                )
                for site, sd in zip(sites_visited_result, sd_result):
                    fh.write(f"{str(site)},{str(sd)}\n")
                fh.close()

            # 紀錄花費時間

            # 繪製site直方圖
            plot_site_detail = {
                "title": "Site Visit Distribution",
                "xlabel": "Counts",
                "ylabel": "Normalized Constant",
                "filename": f"Site Visit Distribution at step {step}.png",
            }
            self.result_histo_plotting(sites_visited_result, plot_site_detail)

            # 繪製sd直方圖
            plot_sd_detail = {
                "title": "SD Distribution",
                "xlabel": "SD",
                "ylabel": "Counts",
                "filename": f"SD Distribution at step {step}.png",
            }
            self.result_histo_plotting(sd_result, plot_sd_detail)

        os.chdir(self.maindir)
        return
 
class SerialInterdiffusionAnalyzer:
    def __init__(
        self,
        alloy_system,
        repetitions,
        lattice_size,
        plane_length,
        couple_length,
        lattice_record,
        v_position_record,
        vacancy_record,
    ):
        self.alloy_system = alloy_system
        self.lattice_size = lattice_size
        self.plane_length = int(plane_length * self.lattice_size)
        self.couple_length = int(couple_length * self.lattice_size)
        self.plane_atom_num = self.plane_length**2
        self.repetition = int(repetitions)
        self.successtrial = len(lattice_record)
        self.element_type = ["A", "B", "C", "D", "E"]
        self.lattice_record = lattice_record
        self.v_position_record = v_position_record
        self.vacancy_record = vacancy_record
        self.x_index = [col for col in self.vacancy_record.index if "x" in col]
        self.y_index = [col for col in self.vacancy_record.index if "y" in col]
        self.z_index = [col for col in self.vacancy_record.index if "z" in col]
        self.sd_index = [col for col in self.vacancy_record.index if "SD" in col]
        self.maindir = os.path.abspath(os.curdir)

    def extract_each_vacancy_lattice_record(self, realization):
        return self.lattice_record.loc[realization]

    def position_composition(self, lattice):
        """
        根據傳入的Lattice array紀錄, 回傳不同Plane上的不同元素的佔比, 為字典形式
        """
        composition_record = {}
        for i, x in enumerate(
            range(0, 2 * self.couple_length * self.plane_atom_num, self.plane_atom_num)
        ):
            atoms = lattice[x : x + 1 * self.plane_atom_num]
            element, concentration = np.unique(atoms, return_counts=True)
            composition_record[i] = dict(zip(element, concentration / len(atoms)))
        return composition_record

    def vacancy_concentration_profile(self, vacancy):
        record = self.extract_each_vacancy_lattice_record(vacancy)
        concentration_df = pd.DataFrame()
        dfs_to_concat = []
        for i, rep_walker in enumerate(record):
            composition = self.position_composition(rep_walker)
            composition_df = pd.DataFrame.from_dict(composition, orient="index")
            composition_df_sorted = composition_df.sort_index()

            walker_df = pd.DataFrame(index=composition_df_sorted.index)
            for column in composition_df_sorted.columns:
                if column != 0:
                    walker_df[f"{i}_{chr(64+column)}"] = composition_df_sorted[column]
            dfs_to_concat.append(walker_df)
        concentration_df = pd.concat(dfs_to_concat, axis=1).fillna(0)
        result_df = pd.DataFrame(index=composition_df_sorted.index)

        # 將多次結果平均
        for element in self.element_type:
            columns_with_element = concentration_df.columns[
                concentration_df.columns.str.contains(f"_{element}")
            ]
            if columns_with_element.any():
                result_df[element] = concentration_df[columns_with_element].mean(axis=1)
                result_df[f"{element}_error"] = concentration_df[
                    columns_with_element
                ].std(axis=1)
        result_df.to_excel(f"Concentration as position of {vacancy}.xlsx")

        return result_df

    def draw_concentration_profile(self, vacancy, concentration_df):
        for element in self.element_type:
            if element in concentration_df.columns:
                plt.plot(
                    concentration_df.index,
                    concentration_df[element],
                    marker="s",
                    linestyle="-",
                    label=element,
                )
        plt.legend()
        plt.title(f"After {vacancy} vacancys.")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"Final After{vacancy} vacancys.jpg")
        plt.close()

    def draw_enlarge_concentration_profile(self, vacancy, concentration_df):
        fig, ax = plt.subplots()
        # Original Plot
        elementlist = []
        for element in self.element_type:
            if element in concentration_df.columns:
                elementlist.append(element)
                ax.plot(
                    concentration_df.index,
                    concentration_df[element],
                    marker="s",
                    linestyle="-",
                    label=element,
                )
        ax.legend()
        ax.set_title(f"After {vacancy} vacancys.")
        ax.set_xlabel("Position")
        ax.set_ylabel("Concentration")

        x1, x2, y1, y2 = (
            self.couple_length - 15,
            self.couple_length + 15,
            min(concentration_df[elementlist].min()) - 0.1,
            max(concentration_df[elementlist].max()) + 0.1,
        )  # subregion of the original image
        axins = ax.inset_axes(
            [0.55, 0.55, 0.3, 0.3],
            xlim=(x1, x2),
            ylim=(y1, y2),
            xticklabels=[],
            yticklabels=[],
        )
        # Enlarged Plot
        for element in self.element_type:
            if element in concentration_df.columns:
                axins.errorbar(
                    concentration_df.index,
                    concentration_df[element],
                    yerr=concentration_df[f"{element}_error"],
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    label=element,
                )
        ax.indicate_inset_zoom(axins, edgecolor="black")
        # Adjust the layout for the entire figure
        plt.tight_layout()
        plt.title(f"After {vacancy} vacancys")
        # Show the figure
        plt.savefig(f"Enlarge inside After {vacancy} vacancys.jpg")
        # plt.show()
        plt.close()
        return

    def draw_continous_concentration_profile(
        self,
    ):
        os.chdir("concentrationresult")
        plot_detail = {
            "linestyle": {
                0: "-",
                1: (0, (1, 10)),
                2: ":",
                3: (0, (5, 1)),
                4: "--",
                5: "-.",
            },
            "marker": {
                0: "o",
                1: "s",
                2: "v",
                3: "p",
                4: "D",
                5: "*",
            },
            "color": {
                "A1A2": ["black", "red"],
                "AB1AB2": ["black", "red"],
                "ABC1ABC2": ["black", "green", "red"],
                "ABCD1ABCD2": ["black", "green", "yellow", "red"],
                "ABCDE1ABCDE2": ["black", "blue", "green", "yellow", "red"],
            },
        }
        for i, numvacancy in enumerate([0, 10, 30, 50, 80, 100]):
            # Convert lattice information into concetration as function of position
            concentration_df = self.vacancy_concentration_profile(numvacancy)
            # Plot whole concentration profile without/ with enlarge plot
            for j, element in enumerate(self.element_type):
                if element in concentration_df.columns:
                    plt.plot(
                        concentration_df.index[
                            self.couple_length - 12 : self.couple_length + 12
                        ],
                        concentration_df[element][
                            self.couple_length - 12 : self.couple_length + 12
                        ],
                        color=plot_detail["color"][self.alloy_system][j],
                        # linestyle=plot_detail["linestyle"][i],
                        marker=plot_detail["marker"][i],
                        label=element,
                    )
        plt.title("Concentraion Profile evolution with # of vacancy")
        plt.ylabel("Concentration")
        plt.xlabel("Position")
        plt.savefig(f"Concentraion Profile evolution with # of vacancy.png")
        plt.close()
        os.chdir(self.maindir)
        return

    def vacancy_concentration_analysis(self):
        if not os.path.exists(r"concentrationresult"):
            os.makedirs(r"concentrationresult")
        os.chdir("concentrationresult")
        for numvacancy in [0, 10, 30, 50, 80, 100]:
            # Convert lattice information into concetration as function of position
            concentration_df = self.vacancy_concentration_profile(numvacancy)
            # Plot whole concentration profile without/ with enlarge plot
            self.draw_concentration_profile(numvacancy, concentration_df)
            self.draw_enlarge_concentration_profile(numvacancy, concentration_df)
            # Plot derivative of concentration profile, corresponding to flux and accumulation
        os.chdir(self.maindir)
        return

    def run_concentration_analysis(self):
        print(f"Now processing analysis of concentration ... ")
        self.vacancy_concentration_analysis()
        self.draw_continous_concentration_profile()

    def num_vacancy_result_plotting(self, result_df, plot_detail):
        result_df["Mean"] = result_df.mean(axis=1)
        plt.scatter(result_df.index, result_df["Mean"])
        plt.title(plot_detail["title"])
        plt.xlabel(plot_detail["xlabel"])
        plt.ylabel(plot_detail["ylabel"])
        plt.tight_layout()
        plt.savefig(plot_detail["filename"])
        plt.close()
        return

    def vacancy_backhoppro_analysis(self):
        print(f"The task is currently in progress of vacancy backhop probability...")
        if not os.path.exists("simulationresult"):
            os.makedirs("simulationresult")
        os.chdir("simulationresult")
        backhop_probability_record = {}
        for trial in self.vacancy_record.columns:
            backhop_probability_record[trial] = {}
            for numofvacancy, (
                numofvacancy_x,
                numofvacancy_y,
                numofvacancy_z,
            ) in enumerate(zip(self.x_index, self.y_index, self.z_index)):
                record = self.vacancy_record.loc[
                    [numofvacancy_x, numofvacancy_y, numofvacancy_z], trial
                ]
                record_df = pd.DataFrame(
                    record.tolist(), index=record.index
                ).transpose()
                df = pd.DataFrame()
                difference = (
                    record_df[[numofvacancy_x, numofvacancy_y, numofvacancy_z]]
                    .diff()
                    .fillna(value=record_df.iloc[[0]])
                )
                shift = difference.shift(periods=1).fillna(0)
                df[numofvacancy_x] = difference[numofvacancy_x] + shift[numofvacancy_x]
                df[numofvacancy_y] = difference[numofvacancy_y] + shift[numofvacancy_y]
                df[numofvacancy_z] = difference[numofvacancy_z] + shift[numofvacancy_z]
                backhop_probability_record[trial][numofvacancy] = (
                    df[
                        (df[numofvacancy_x] == 0)
                        & (df[numofvacancy_y] == 0)
                        & (df[numofvacancy_z] == 0)
                    ].shape[0]
                    / df.shape[0]
                )

        backhop_probability_result_df = pd.DataFrame(backhop_probability_record)
        backhop_probability_result_df.to_json(f"backhopprobability.json")
        # 分析不同VACANCY的變化
        plot_backhoppro_detail = {
            "title": "Back hop probability of different Vacancy",
            "xlabel": "vacancy number",
            "ylabel": "Probability",
            "filename": f"Backhop probability.png",
        }
        self.num_vacancy_result_plotting(
            backhop_probability_result_df, plot_backhoppro_detail
        )
        os.chdir(self.maindir)
        return

    def vacancy_sitesd_analysis(self):
        print(f"The task is currently in progress of vacancy site sd recording...")
        if not os.path.exists("simulationresult"):
            os.makedirs("simulationresult")
        os.chdir("simulationresult")
        site_result = {}
        sd_result = {}
        for trial in self.vacancy_record.columns:
            # 第幾次取樣結果
            site_result[trial] = {}
            sd_result[trial] = {}
            for numofvacancy, (
                numofvacancy_x,
                numofvacancy_y,
                numofvacancy_z,
                numofvacancysd,
            ) in enumerate(
                zip(self.x_index, self.y_index, self.z_index, self.sd_index)
            ):
                # 第n顆vacancy結果
                record = self.vacancy_record.loc[
                    [numofvacancy_x, numofvacancy_y, numofvacancy_z], trial
                ]
                df = pd.DataFrame(record.tolist(), index=record.index).transpose()
                df["count"] = 1
                site_couts = (
                    df.groupby([numofvacancy_x, numofvacancy_y, numofvacancy_z])
                    .agg({"count": len})
                    .reset_index()
                )
                num_sites_visited = site_couts.shape[0]
                site_result[trial][numofvacancy] = num_sites_visited
                sd_result[trial][numofvacancy] = self.vacancy_record.at[
                    numofvacancysd, trial
                ][-1]

        # 分析 site 隨 vacancy 變化
        site_result_df = pd.DataFrame(site_result)
        plot_site_detail = {
            "title": "Site visited num of different Vacancy",
            "xlabel": "vacancy number",
            "ylabel": "# of sites",
            "filename": f"Site visit.png",
        }
        self.num_vacancy_result_plotting(site_result_df, plot_site_detail)
        site_result_df.to_json(f"site result.json")

        # 分析 sd 隨 vacancy 變化
        sd_result_df = pd.DataFrame(sd_result)
        plot_sd_detail = {
            "title": "MSD of different Vacancy",
            "xlabel": "vacancy number",
            "ylabel": "MSD",
            "filename": f"Sd.png",
        }
        self.num_vacancy_result_plotting(sd_result_df, plot_sd_detail)
        sd_result_df.to_json(f"sd result.json")
        os.chdir(self.maindir)
        return

    def vacancy_steptoIDZ_analysis(self):
        print(
            f"The task is currently in progress of vacancy step to interdiffusion zone boundary..."
        )
        if not os.path.exists("simulationresult"):
            os.makedirs("simulationresult")
        os.chdir("simulationresult")
        step_record = {}
        for trial in self.vacancy_record.columns:
            step_record[trial] = {}
            for numofvacancy, (
                numofvacancy_x,
                numofvacancy_y,
                numofvacancy_z,
            ) in enumerate(zip(self.x_index, self.y_index, self.z_index)):
                record = self.vacancy_record.loc[
                    [numofvacancy_x, numofvacancy_y, numofvacancy_z], trial
                ]
                record_df = pd.DataFrame(
                    record.tolist(), index=record.index
                ).transpose()

                step_record[trial][numofvacancy] = record_df.shape[0]

        step_result_df = pd.DataFrame(step_record)
        step_result_df.to_json(f"steptoIDZ.json")
        # 分析不同VACANCY的變化
        plot_step_detail = {
            "title": "Step to interdiffusion zone boundary of different Vacancy",
            "xlabel": "vacancy number",
            "ylabel": "# of steps",
            "filename": f"Step to zone boundary.png",
        }
        self.num_vacancy_result_plotting(step_result_df, plot_step_detail)
        os.chdir(self.maindir)
        return

    def run_vacancy_behavior_analysis(self):
        self.vacancy_backhoppro_analysis()
        self.vacancy_sitesd_analysis()
        self.vacancy_steptoIDZ_analysis()
        return

class ErfInterdiffusionAnalyzer:
    def __init__(
        self,
        alloy_system,
        repetitions,
        lattice_size,
        plane_length,
        couple_length,
        lattice_record,
        v_position_record,
        vacancy_record,
    ):
        self.alloy_system = alloy_system
        self.lattice_size = lattice_size
        self.plane_length = int(plane_length * self.lattice_size)
        self.couple_length = int(couple_length * self.lattice_size)
        self.plane_atom_num = self.plane_length**2
        self.repetition = int(repetitions)
        self.successtrial = len(lattice_record)
        self.recordstep = [10, 30, 50]
        self.element_type = ["A", "B", "C", "D", "E"]
        self.lattice_record = lattice_record
        self.v_position_record = v_position_record
        self.vacancy_record = vacancy_record
        self.x_index = [col for col in self.vacancy_record.index if "x" in col]
        self.y_index = [col for col in self.vacancy_record.index if "y" in col]
        self.z_index = [col for col in self.vacancy_record.index if "z" in col]
        self.sd_index = [col for col in self.vacancy_record.index if "SD" in col]
        self.maindir = os.path.abspath(os.curdir)

    def position_composition(self, lattice):
        """
        根據傳入的Lattice array紀錄, 回傳不同Plane上的不同元素的佔比, 為字典形式
        """
        composition_record = {}
        for i, x in enumerate(
            range(0, 2 * self.couple_length * self.plane_atom_num, self.plane_atom_num)
        ):
            atoms = lattice[x : x + 1 * self.plane_atom_num]
            element, concentration = np.unique(atoms, return_counts=True)
            composition_record[i] = dict(zip(element, concentration / len(atoms)))
        return composition_record

    def vacancy_concentration_profile(self):
        concentration_df = pd.DataFrame()
        dfs_to_concat = []
        for i, rep in enumerate(self.lattice_record.columns):
            lattice = np.array(self.lattice_record[rep].values.flatten())
            composition = self.position_composition(lattice)
            composition_df = pd.DataFrame.from_dict(composition, orient="index")
            composition_df_sorted = composition_df.sort_index()

            walker_df = pd.DataFrame(index=composition_df_sorted.index)
            for column in composition_df_sorted.columns:
                if column != 0:
                    walker_df[f"{i}_{chr(64+column)}"] = composition_df_sorted[column]
            dfs_to_concat.append(walker_df)

        concentration_df = pd.concat(dfs_to_concat, axis=1)
        result_df = pd.DataFrame(index=concentration_df.index)
        for element in self.element_type:
            columns_with_element = concentration_df.columns[
                concentration_df.columns.str.contains(f"_{element}")
            ]
            if columns_with_element.any():
                result_df[element] = concentration_df[columns_with_element].mean(axis=1)
                result_df[f"{element}_error"] = concentration_df[
                    columns_with_element
                ].std(axis=1)
        result_df.to_excel(f"Concentration as position.xlsx")
        return result_df

    def draw_concentration_profile(self, concentration_df):
        for element in self.element_type:
            if element in concentration_df.columns:
                plt.plot(
                    concentration_df.index,
                    concentration_df[element],
                    marker="s",
                    linestyle="-",
                    label=element,
                )
        plt.legend()
        plt.xlabel("Position")
        plt.title(f"Concentration Profile of {self.alloy_system}")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"Concentration Profile of {self.alloy_system}.jpg")
        plt.close()
        return

    def vacancy_concentration_analysis(self):
        if not os.path.exists(r"concentrationresult"):
            os.makedirs(r"concentrationresult")
        os.chdir("concentrationresult")
        # Convert lattice information into concetration as function of position
        concentration_df = self.vacancy_concentration_profile()
        # Plot whole concentration profile without/ with enlarge plot
        self.draw_concentration_profile(concentration_df)
        os.chdir(self.maindir)
        return

    def vacancy_position_distribution(self):
        if not os.path.exists(r"distribution"):
            os.makedirs(r"distribution")
        os.chdir("distribution")
        distribution_df = pd.DataFrame()
        for i, step in enumerate(self.recordstep):

            position_layer = self.v_position_record.applymap(
                lambda x: x[i] // self.plane_atom_num
            ).values.flatten()

            df = pd.Series(position_layer)
            distribution_df[step] = df
        # 繪製
        position = np.arange(self.couple_length * 2)
        # hist plot
        for step in distribution_df.columns:
            plt.hist(distribution_df[step], bins=position, density=True, alpha=0.8)
        plt.title(f"{self.alloy_system}")
        plt.xlabel("Postiion")
        plt.savefig(f"Vacancy distribution of {self.alloy_system}.jpg")
        # plt.show()
        plt.close()

        # 繪製同位置隨step的變化
        plt.hist(
            distribution_df,
            bins=np.arange(0, 20),
            histtype="bar",
            label=distribution_df.columns,
        )
        plt.xlabel("Position")
        plt.xticks(np.arange(0, 20))
        plt.ylabel("Counts")
        plt.title(f"Postion Distribution as steps evolution")
        plt.legend()
        plt.savefig(f"Position Distribution as steps evolution.png")
        plt.close()

        distribution_df.to_json(f"Vacancy distribution of {self.alloy_system}.json")
        os.chdir(self.maindir)
        return

    def concentration_distribution(self):
        # read cocnentration profile
        os.chdir(os.path.join(self.maindir, "concentrationresult"))
        concentration = pd.read_excel("Concentration as position.xlsx")

        # read vacancy position distribution
        os.chdir(os.path.join(self.maindir, "distribution"))
        distribution = pd.read_json(f"Vacancy distribution of {self.alloy_system}.json")

        os.chdir(os.path.join(self.maindir, "simulationresult"))

        stepcolor = ["black", "purple", "blue", "green", "yellow", "orange", "red"]
        # Plot together
        fig, ax1 = plt.subplots()
        for element in self.element_type:
            if element in concentration.columns:
                ax1.plot(
                    concentration.index,
                    concentration[element],
                    marker="s",
                    linestyle="-",
                    label=element,
                )
        ax1.set_ylabel("Concentration")

        ax2 = ax1.twinx()
        position = np.arange(self.couple_length * 2)
        # hist plot
        for i, step in enumerate(distribution.columns):
            ax2.hist(
                distribution[step],
                bins=position,
                histtype="step",
                density=True,
                alpha=0.5,
                label=step,
            )
        ax2.set_ylabel("Probability")
        ax2.set_xlabel("Position")
        plt.xticks(position)
        plt.legend()
        plt.title(f"Concentration and Vacancy distribution")
        plt.tight_layout()
        plt.savefig(f"Concentration and Vacancy distribution.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def run_concentration_analysis(self):
        print(f"Now processing analysis of concentration ... ")
        self.vacancy_concentration_analysis()

    def num_vacancy_result_plotting(self, result_df, plot_detail):
        result_df["Mean"] = result_df.mean(axis=1)
        plt.scatter(result_df.index, result_df["Mean"])
        plt.title(plot_detail["title"])
        plt.xlabel(plot_detail["xlabel"])
        plt.ylabel(plot_detail["ylabel"])
        plt.tight_layout()
        plt.savefig(plot_detail["filename"])
        plt.close()
        return

    def vacancy_step_backhoppro_analysis(self):
        print(f"The task is currently in progress of vacancy backhop probability...")
        if not os.path.exists("simulationresult"):
            os.makedirs("simulationresult")
        os.chdir("simulationresult")
        step_backhop_probability_record = {step: [] for step in self.recordstep}
        for trial in self.vacancy_record.columns:
            for numofvacancy, (
                numofvacancy_x,
                numofvacancy_y,
                numofvacancy_z,
            ) in enumerate(zip(self.x_index, self.y_index, self.z_index)):
                record = self.vacancy_record.loc[
                    [numofvacancy_x, numofvacancy_y, numofvacancy_z], trial
                ]
                record_df = pd.DataFrame(
                    record.tolist(), index=record.index
                ).transpose()
                df = pd.DataFrame()
                difference = (
                    record_df[[numofvacancy_x, numofvacancy_y, numofvacancy_z]]
                    .diff()
                    .fillna(value=record_df.iloc[[0]])
                )
                shift = difference.shift(periods=1).fillna(0)
                df[numofvacancy_x] = difference[numofvacancy_x] + shift[numofvacancy_x]
                df[numofvacancy_y] = difference[numofvacancy_y] + shift[numofvacancy_y]
                df[numofvacancy_z] = difference[numofvacancy_z] + shift[numofvacancy_z]
                for step in self.recordstep:
                    step_df = df[:step]
                    step_backhop_probability_record[step].append(
                        step_df[
                            (step_df[numofvacancy_x] == 0)
                            & (step_df[numofvacancy_y] == 0)
                            & (step_df[numofvacancy_z] == 0)
                        ].shape[0]
                        / step_df.shape[0]
                    )
        step_backhop_probability_record_df = pd.DataFrame.from_dict(
            step_backhop_probability_record
        )
        plt.hist(
            step_backhop_probability_record_df,
            label=step_backhop_probability_record_df.columns,
        )
        plt.title(f"backhop probability distribution with different steps")
        plt.xlabel("Back-hop probability")
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig(f"backhop probability distribution with different steps.jpg")
        plt.close()
        step_backhop_probability_record_df.to_json(f"stepbackhoppro.json")
        os.chdir(self.maindir)
        return

    def vacancy_step_sitesd_analysis(self):
        print(f"The task is currently in progress of vacancy site sd recording...")
        if not os.path.exists("simulationresult"):
            os.makedirs("simulationresult")
        os.chdir("simulationresult")
        site_result = {step: [] for step in self.recordstep}
        sd_result = {step: [] for step in self.recordstep}
        for trial in self.vacancy_record.columns:
            # 第幾次取樣結果
            for numofvacancy, (
                numofvacancy_x,
                numofvacancy_y,
                numofvacancy_z,
                numofvacancysd,
            ) in enumerate(
                zip(self.x_index, self.y_index, self.z_index, self.sd_index)
            ):
                # 第n顆vacancy結果
                record = self.vacancy_record.loc[
                    [numofvacancy_x, numofvacancy_y, numofvacancy_z], trial
                ]
                df = pd.DataFrame(record.tolist(), index=record.index).transpose()
                for step in self.recordstep:
                    step_df = df[:step].copy()
                    step_df["count"] = 1
                    site_couts = (
                        step_df.groupby(
                            [numofvacancy_x, numofvacancy_y, numofvacancy_z]
                        )
                        .agg({"count": len})
                        .reset_index()
                    )
                    num_sites_visited = site_couts.shape[0]
                    site_result[step].append(num_sites_visited)
                    sd_result[step].append(
                        self.vacancy_record.at[numofvacancysd, trial][step - 1]
                    )

        # 分析 site 隨 step 變化
        site_result_df = pd.DataFrame.from_dict(site_result)
        plt.hist(site_result_df, label=site_result_df.columns)
        plt.title(f"Site istribution with different steps")
        plt.xlabel("# of sites")
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig(f"Site with different steps.jpg")
        plt.close()
        site_result_df.to_json(f"stepsiteresult.json")

        # 分析 site 隨 vacancy 變化
        sd_result_df = pd.DataFrame.from_dict(sd_result)
        plt.hist(sd_result_df, label=sd_result_df.columns)
        plt.title(f"SD with different steps")
        plt.xlabel("SD")
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig(f"SD distribution with different steps.jpg")
        plt.close()
        sd_result_df.to_json(f"stepsdresult.json")
        os.chdir(self.maindir)
        return

    def run_vacancy_behavior_analysis(self):
        self.vacancy_position_distribution()
        self.vacancy_step_backhoppro_analysis()
        self.vacancy_step_sitesd_analysis()
        return

class GradientInterdiffusionAnalyzer:
    def __init__(
        self,
        alloy_system,
        repetitions,
        lattice_size,
        plane_length,
        couple_length,
        lattice_record,
        v_position_record,
        vacancy_record,
    ):
        self.alloy_system = alloy_system
        self.lattice_size = lattice_size
        self.plane_length = int(plane_length * self.lattice_size)
        self.couple_length = int(couple_length * self.lattice_size)
        self.plane_atom_num = self.plane_length**2
        self.repetition = int(repetitions)
        self.successtrial = len(lattice_record)
        self.recordstep = [10, 20, 30, 40, 50]
        self.element_type = ["A", "B", "C", "D", "E"]
        self.lattice_record = lattice_record
        self.v_position_record = v_position_record
        self.vacancy_record = vacancy_record
        self.x_index = [col for col in self.vacancy_record.index if "x" in col]
        self.y_index = [col for col in self.vacancy_record.index if "y" in col]
        self.z_index = [col for col in self.vacancy_record.index if "z" in col]
        self.sd_index = [col for col in self.vacancy_record.index if "SD" in col]
        self.maindir = os.path.abspath(os.curdir)

    def position_composition(self, lattice):
        """
        根據傳入的Lattice array紀錄, 回傳不同Plane上的不同元素的佔比, 為字典形式
        """
        composition_record = {}
        for i, x in enumerate(
            range(0, 2 * self.couple_length * self.plane_atom_num, self.plane_atom_num)
        ):
            atoms = lattice[x : x + 1 * self.plane_atom_num]
            element, concentration = np.unique(atoms, return_counts=True)
            composition_record[i] = dict(zip(element, concentration / len(atoms)))
        return composition_record

    def vacancy_concentration_profile(self):
        concentration_df = pd.DataFrame()
        dfs_to_concat = []
        for i, rep in enumerate(self.lattice_record.columns):
            lattice = np.array(self.lattice_record[rep].values.flatten())
            composition = self.position_composition(lattice)
            composition_df = pd.DataFrame.from_dict(composition, orient="index")
            composition_df_sorted = composition_df.sort_index()

            walker_df = pd.DataFrame(index=composition_df_sorted.index)
            for column in composition_df_sorted.columns:
                if column != 0:
                    walker_df[f"{i}_{chr(64+column)}"] = composition_df_sorted[column]
            dfs_to_concat.append(walker_df)

        concentration_df = pd.concat(dfs_to_concat, axis=1)
        result_df = pd.DataFrame(index=concentration_df.index)
        for element in self.element_type:
            columns_with_element = concentration_df.columns[
                concentration_df.columns.str.contains(f"_{element}")
            ]
            if columns_with_element.any():
                result_df[element] = concentration_df[columns_with_element].mean(axis=1)
                result_df[f"{element}_error"] = concentration_df[
                    columns_with_element
                ].std(axis=1)
        result_df.to_excel(f"Concentration as position.xlsx")
        return result_df

    def draw_concentration_profile(self, concentration_df):
        for element in self.element_type:
            if element in concentration_df.columns:
                plt.plot(
                    concentration_df.index,
                    concentration_df[element],
                    marker="s",
                    linestyle="-",
                    label=element,
                )
        plt.legend()
        plt.xlabel("Position")
        plt.title(f"Concentration Profile of {self.alloy_system}")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"Concentration Profile of {self.alloy_system}.jpg")
        plt.close()
        return

    def vacancy_concentration_analysis(self):
        if not os.path.exists(r"concentrationresult"):
            os.makedirs(r"concentrationresult")
        os.chdir("concentrationresult")
        # Convert lattice information into concetration as function of position
        concentration_df = self.vacancy_concentration_profile()
        # Plot whole concentration profile without/ with enlarge plot
        self.draw_concentration_profile(concentration_df)
        os.chdir(self.maindir)
        return

    def vacancy_position_distribution(self):
        if not os.path.exists(r"distribution"):
            os.makedirs(r"distribution")
        os.chdir("distribution")
        distribution_df = pd.DataFrame()
        for i, step in enumerate(self.recordstep):

            position_layer = self.v_position_record.applymap(
                lambda x: x[i] // self.plane_atom_num
            ).values.flatten()

            df = pd.Series(position_layer)
            distribution_df[step] = df
        # 繪製
        position = np.arange(self.couple_length * 2)
        # hist plot
        for step in distribution_df.columns:
            plt.hist(distribution_df[step], bins=position, density=True, alpha=0.8)
        plt.title(f"{self.alloy_system}")
        plt.xlabel("Postiion")
        plt.savefig(f"Vacancy distribution of {self.alloy_system}.jpg")
        # plt.show()
        plt.close()

        # 繪製同位置隨step的變化
        plt.hist(
            distribution_df,
            bins=np.arange(0, 20),
            histtype="bar",
            label=distribution_df.columns,
        )
        plt.xlabel("Position")
        plt.xticks(np.arange(0, 20))
        plt.ylabel("Counts")
        plt.title(f"Postion Distribution as steps evolution")
        plt.legend()
        plt.savefig(f"Position Distribution as steps evolution.png")
        plt.close()

        distribution_df.to_json(f"Vacancy distribution of {self.alloy_system}.json")
        os.chdir(self.maindir)
        return

    def concentration_distribution(self):
        # read cocnentration profile
        os.chdir(os.path.join(self.maindir, "concentrationresult"))
        concentration = pd.read_excel("Concentration as position.xlsx")

        # read vacancy position distribution
        os.chdir(os.path.join(self.maindir, "distribution"))
        distribution = pd.read_json(f"Vacancy distribution of {self.alloy_system}.json")

        os.chdir(os.path.join(self.maindir, "simulationresult"))

        stepcolor = ["black", "purple", "blue", "green", "yellow", "orange", "red"]
        # Plot together
        fig, ax1 = plt.subplots()
        for element in self.element_type:
            if element in concentration.columns:
                ax1.plot(
                    concentration.index,
                    concentration[element],
                    marker="s",
                    linestyle="-",
                    label=element,
                )
        ax1.set_ylabel("Concentration")

        ax2 = ax1.twinx()
        position = np.arange(self.couple_length * 2)
        # hist plot
        for i, step in enumerate(distribution.columns):
            ax2.hist(
                distribution[step],
                bins=position,
                histtype="step",
                density=True,
                alpha=0.5,
                label=step,
            )
        ax2.set_ylabel("Probability")
        ax2.set_xlabel("Position")
        plt.xticks(position)
        plt.legend()
        plt.title(f"Concentration and Vacancy distribution")
        plt.tight_layout()
        plt.savefig(f"Concentration and Vacancy distribution.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def run_concentration_analysis(self):
        print(f"Now processing analysis of concentration ... ")
        self.vacancy_concentration_analysis()

    def vacancy_step_backhoppro_analysis(self):
        print(f"The task is currently in progress of vacancy backhop probability...")
        if not os.path.exists("simulationresult"):
            os.makedirs("simulationresult")
        os.chdir("simulationresult")
        step_backhop_probability_record = {step: [] for step in self.recordstep}
        for trial in self.vacancy_record.columns:
            for numofvacancy, (
                numofvacancy_x,
                numofvacancy_y,
                numofvacancy_z,
            ) in enumerate(zip(self.x_index, self.y_index, self.z_index)):
                record = self.vacancy_record.loc[
                    [numofvacancy_x, numofvacancy_y, numofvacancy_z], trial
                ]
                record_df = pd.DataFrame(
                    record.tolist(), index=record.index
                ).transpose()
                df = pd.DataFrame()
                difference = (
                    record_df[[numofvacancy_x, numofvacancy_y, numofvacancy_z]]
                    .diff()
                    .fillna(value=record_df.iloc[[0]])
                )
                shift = difference.shift(periods=1).fillna(0)
                df[numofvacancy_x] = difference[numofvacancy_x] + shift[numofvacancy_x]
                df[numofvacancy_y] = difference[numofvacancy_y] + shift[numofvacancy_y]
                df[numofvacancy_z] = difference[numofvacancy_z] + shift[numofvacancy_z]
                for step in self.recordstep:
                    step_df = df[:step]
                    step_backhop_probability_record[step].append(
                        step_df[
                            (step_df[numofvacancy_x] == 0)
                            & (step_df[numofvacancy_y] == 0)
                            & (step_df[numofvacancy_z] == 0)
                        ].shape[0]
                        / step_df.shape[0]
                    )
        step_backhop_probability_record_df = pd.DataFrame.from_dict(
            step_backhop_probability_record
        )
        plt.hist(
            step_backhop_probability_record_df,
            label=step_backhop_probability_record_df.columns,
        )
        plt.title(f"backhop probability distribution with different steps")
        plt.xlabel("Back-hop probability")
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig(f"backhop probability distribution with different steps.jpg")
        plt.close()
        step_backhop_probability_record_df.to_json(f"stepbackhoppro.json")
        os.chdir(self.maindir)
        return

    def vacancy_step_sitesd_analysis(self):
        print(f"The task is currently in progress of vacancy site sd recording...")
        if not os.path.exists("simulationresult"):
            os.makedirs("simulationresult")
        os.chdir("simulationresult")
        site_result = {step: [] for step in self.recordstep}
        sd_result = {step: [] for step in self.recordstep}
        for trial in self.vacancy_record.columns:
            # 第幾次取樣結果
            for numofvacancy, (
                numofvacancy_x,
                numofvacancy_y,
                numofvacancy_z,
                numofvacancysd,
            ) in enumerate(
                zip(self.x_index, self.y_index, self.z_index, self.sd_index)
            ):
                # 第n顆vacancy結果
                record = self.vacancy_record.loc[
                    [numofvacancy_x, numofvacancy_y, numofvacancy_z], trial
                ]
                df = pd.DataFrame(record.tolist(), index=record.index).transpose()
                for step in self.recordstep:
                    step_df = df[:step].copy()
                    step_df["count"] = 1
                    site_couts = (
                        step_df.groupby(
                            [numofvacancy_x, numofvacancy_y, numofvacancy_z]
                        )
                        .agg({"count": len})
                        .reset_index()
                    )
                    num_sites_visited = site_couts.shape[0]
                    site_result[step].append(num_sites_visited)
                    sd_result[step].append(
                        self.vacancy_record.at[numofvacancysd, trial][step - 1]
                    )

        # 分析 site 隨 step 變化
        site_result_df = pd.DataFrame.from_dict(site_result)
        plt.hist(site_result_df, label=site_result_df.columns)
        plt.title(f"Site istribution with different steps")
        plt.xlabel("# of sites")
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig(f"Site with different steps.jpg")
        plt.close()
        site_result_df.to_json(f"stepsiteresult.json")

        # 分析 site 隨 vacancy 變化
        sd_result_df = pd.DataFrame.from_dict(sd_result)
        plt.hist(sd_result_df, label=sd_result_df.columns)
        plt.title(f"SD with different steps")
        plt.xlabel("SD")
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig(f"SD distribution with different steps.jpg")
        plt.close()
        sd_result_df.to_json(f"stepsdresult.json")
        os.chdir(self.maindir)
        return

    def run_vacancy_behavior_analysis(self):
        self.vacancy_position_distribution()
        self.vacancy_step_backhoppro_analysis()
        self.vacancy_step_sitesd_analysis()
        return

class SerialAnalyzer:
    def __init__(self, filepath):
        self.crystalstructure = ["sc", "bcc", "fcc"]
        self.flucsystem = [
            "serial_dynamicIDZ_diffpro_data_size10_numofvacancy100_nb2.5_5fluc",
            "serial_dynamicIDZ_diffpro_data_size10_numofvacancy100_wb5_5fluc",
            "serial_dynamicIDZ_diffpro_data_size10_numofvacancy100_nb2.5_10fluc",
            "serial_dynamicIDZ_diffpro_data_size10_numofvacancy100_wb5_10fluc",
        ]
        self.fliepath = filepath
        self.fluctype = [
            "nb2.5_5fluc",
            "wb5_5fluc",
            "nb2.5_10fluc",
            "wb5_10fluc",
        ]
        self.alloytype = ["ABp", "AB", "ABC", "ABCD", "ABCDE"]
        self.alloycolor = ["black", "blue", "green", "orange", "red"]
        self.fluccolor = ["black", "blue", "red", "green"]
        self.element_type = ["A", "B", "C", "D", "E"]
        self.vacancygroup = [0, 10, 30, 50, 80, 100]
        self.groupmarker = ["o", "s", "v", "p", "D", "*"]
        self.alloysys = ["A1A2", "AB1AB2", "ABC1ABC2", "ABCD1ABCD2", "ABCDE1ABCDE2"]
        self.maindir = filepath
        if not os.path.exists(os.path.join(self.maindir, "ComparisonResult")):
            os.mkdir(os.path.join(self.maindir, "ComparisonResult"))
        for crystal in self.crystalstructure:
            if not os.path.exists(
                os.path.join(self.maindir, "ComparisonResult", f"{crystal}")
            ):
                os.mkdir(os.path.join(self.maindir, "ComparisonResult", f"{crystal}"))
        self.outputdir = os.path.join(self.maindir, "ComparisonResult")
        return

    def alloysystem_idz_range(self):
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                os.chdir(os.path.join(serial_file_path, fluc))
                leftrecord = pd.DataFrame()
                rightrecord = pd.DataFrame()
                for alloytype, alloy in zip(self.alloytype, self.alloysys):
                    os.chdir(os.path.join(serial_file_path, fluc, alloy))
                    idzrecord = pd.read_json(
                        f"idz of {alloy} with 100 json.gz", compression="gzip"
                    )
                    leftidz = idzrecord.iloc[::2, :].mean(axis=1).reset_index(drop=True)
                    rightidz = (
                        idzrecord.iloc[1::2, :].mean(axis=1).reset_index(drop=True)
                    )
                    leftrecord[alloytype] = leftidz - 100
                    rightrecord[alloytype] = rightidz - 100

                if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}"))
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                rightrecord.plot(ax=ax1, color=self.alloycolor)
                leftrecord.plot(ax=ax2, color=self.alloycolor, legend=False)
                ax1.set_title(f"IDZ change as # of V passed under {flucname}")
                ax1.set_ylabel("Right IDZ layer")
                ax1.set_ylim(0, 11)
                ax2.set_ylabel("Left IDZ layer")
                ax2.set_ylim(-9, 0)
                plt.xlabel("# of vacancy passed into")

                plt.tight_layout()
                fig.subplots_adjust(hspace=0)
                plt.savefig(f"IDZ change as # of V passed under {flucname}.jpg")
                plt.close()

        return

    def fluc_idz_range(self):
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                os.chdir(os.path.join(serial_file_path, fluc))
                leftrecord = pd.DataFrame()
                rightrecord = pd.DataFrame()
                for alloytype, alloy in zip(self.alloytype, self.alloysys):
                    os.chdir(os.path.join(serial_file_path, fluc, alloy))
                    idzrecord = pd.read_json(
                        f"idz of {alloy} with 100 json.gz", compression="gzip"
                    )
                    leftidz = idzrecord.iloc[::2, :].mean(axis=1).reset_index(drop=True)
                    rightidz = (
                        idzrecord.iloc[1::2, :].mean(axis=1).reset_index(drop=True)
                    )
                    leftrecord[alloytype] = leftidz - 100
                    rightrecord[alloytype] = rightidz - 100

                if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}"))
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                rightrecord.plot(ax=ax1, color=self.alloycolor)
                leftrecord.plot(ax=ax2, color=self.alloycolor, legend=False)
                ax1.set_title(f"IDZ change as # of V passed under {flucname}")
                ax1.set_ylabel("Right IDZ layer")
                ax1.set_ylim(0, 11)
                ax2.set_ylabel("Left IDZ layer")
                ax2.set_ylim(-9, 0)
                plt.xlabel("# of vacancy passed into")

                plt.tight_layout()
                fig.subplots_adjust(hspace=0)
                plt.savefig(f"IDZ change as # of V passed under {flucname}.jpg")
                plt.close()

        return

    def alloysystem_backhoppro_comparison(self):
        pos = [10, 35, 60, 90]
        group = [0, 20, 50, 80, 100]  # divide into four group based on observation
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                os.chdir(os.path.join(serial_file_path, fluc))
                groupdf = pd.DataFrame(index=pos)
                fig, ax1 = plt.subplots()

                # Each vacancy
                for i, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    hoppro = pd.read_json("backhopprobability.json")
                    result = hoppro.mean(axis=1)
                    ax1.scatter(
                        result.index,
                        result,
                        label=self.alloytype[i],
                        color=self.alloycolor[i],
                    )
                    grouppro = []
                    for j in range(len(group) - 1):
                        grouppro.append(result.iloc[group[j] : group[j + 1]].mean())
                    groupdf[alloy] = grouppro

                ax2 = ax1.twinx()
                # Group
                for k, numofvacancy in enumerate(groupdf.index):
                    for l, (offset, alloy) in enumerate(
                        zip(range(-5, 5, 2), groupdf.columns)
                    ):
                        ax2.bar(
                            numofvacancy + offset,
                            groupdf.at[numofvacancy, alloy],
                            width=2,
                            color=self.alloycolor[l],
                            alpha=0.5,
                        )

                # Save figure
                if not os.path.exists(
                    os.path.join(self.outputdir, f"{crystal}", "SameFluc")
                ):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}", "SameFluc"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}", "SameFluc"))
                ax1.legend(bbox_to_anchor=(0.85, 0.3))
                ax1.set_xlabel("# of vacancy passed into")
                ax1.set_ylabel("Each Vacancy Backhop Probability")
                ax2.set_ylabel("Each Group Vacancy Backhop Probability")
                plt.title(f"Vacancy Backhop probability under {flucname}")
                plt.tight_layout()
                plt.savefig(f"Vacancy Backhop probability under {flucname}.jpg")
                plt.close()
                os.chdir(self.maindir)

            os.chdir(self.maindir)
        return

    def alloysystem_site_comparison(self):
        # 計算 site
        pos = [5, 20, 40, 65, 90]
        group = [0, 10, 30, 50, 80, 100]
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                os.chdir(os.path.join(serial_file_path, fluc))
                groupdf = pd.DataFrame(index=pos)
                fig, ax1 = plt.subplots()
                # Each
                for i, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    sites = pd.read_json("site result.json")
                    result = sites["Mean"]
                    ax1.scatter(
                        result.index,
                        result,
                        label=self.alloytype[i],
                        color=self.alloycolor[i],
                        alpha=0.5,
                    )
                    groupsites = []
                    for j in range(len(group) - 1):
                        groupsites.append(result.iloc[group[j] : group[j + 1]].mean())
                    groupdf[alloy] = groupsites

                ax2 = ax1.twinx()
                # Group
                for k, numofvacancy in enumerate(groupdf.index):
                    for l, (offset, alloy) in enumerate(
                        zip(range(-5, 5, 2), groupdf.columns)
                    ):
                        ax2.bar(
                            numofvacancy + offset,
                            groupdf.at[numofvacancy, alloy],
                            width=2,
                            color=self.alloycolor[l],
                        )

                if not os.path.exists(
                    os.path.join(self.outputdir, f"{crystal}", "SameFluc")
                ):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}", "SameFluc"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}", "SameFluc"))
                ax1.legend()
                ax1.set_xlabel("# of vacancy passed into")
                ax1.set_ylabel("Each Vacancy Average Unique Sites Visited")
                ax2.set_ylabel("Each Group Average Sites Visited")
                plt.title(f"Vacancy Sites Visited under {flucname}")
                plt.tight_layout()
                plt.savefig(f"Vacancy Sites Visited under {flucname}.jpg")
                plt.close()
                os.chdir(self.maindir)

            os.chdir(self.maindir)
        return

    def alloysystem_sd_comparison(self):
        # 計算 site
        pos = [5, 20, 40, 65, 90]
        group = [0, 10, 30, 50, 80, 100]
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                os.chdir(os.path.join(serial_file_path, fluc))
                groupdf = pd.DataFrame(index=pos)
                fig, ax1 = plt.subplots()
                # Each
                for i, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    sd = pd.read_json("sd result.json")
                    result = sd["Mean"]
                    ax1.scatter(
                        result.index,
                        result,
                        label=self.alloytype[i],
                        color=self.alloycolor[i],
                        alpha=0.5,
                    )
                    groupsites = []
                    for j in range(len(group) - 1):
                        groupsites.append(result.iloc[group[j] : group[j + 1]].mean())
                    groupdf[alloy] = groupsites

                ax2 = ax1.twinx()
                # Group
                for k, numofvacancy in enumerate(groupdf.index):
                    for l, (offset, alloy) in enumerate(
                        zip(range(-5, 5, 2), groupdf.columns)
                    ):
                        ax2.bar(
                            numofvacancy + offset,
                            groupdf.at[numofvacancy, alloy],
                            width=2,
                            color=self.alloycolor[l],
                        )
                if not os.path.exists(
                    os.path.join(self.outputdir, f"{crystal}", "SameFluc")
                ):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}", "SameFluc"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}", "SameFluc"))
                ax1.legend()
                ax1.set_xlabel("# of vacancy passed into")
                ax1.set_ylabel("Each Vacancy MSD")
                ax2.set_ylabel("Each Group MSD")
                plt.title(f"Vacancy MSD under {flucname}")
                plt.tight_layout()
                plt.savefig(f"Vacancy MSD under {flucname}.jpg")
                plt.close()
                os.chdir(self.maindir)

            os.chdir(self.maindir)
        return

    def alloysystem_step_comparison(self):
        # 計算 site
        pos = [5, 20, 40, 65, 90]
        group = [0, 10, 30, 50, 80, 100]
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                os.chdir(os.path.join(serial_file_path, fluc))
                groupdf = pd.DataFrame(index=pos)
                fig, ax1 = plt.subplots()
                # Each
                for i, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step = pd.read_json("steptoIDZ.json")
                    result = step.mean(axis=1)
                    ax1.scatter(
                        result.index,
                        result,
                        label=self.alloytype[i],
                        color=self.alloycolor[i],
                        alpha=0.5,
                    )
                    groupsites = []
                    for j in range(len(group) - 1):
                        groupsites.append(result.iloc[group[j] : group[j + 1]].mean())
                    groupdf[alloy] = groupsites

                ax2 = ax1.twinx()
                # Group
                for k, numofvacancy in enumerate(groupdf.index):
                    for l, (offset, alloy) in enumerate(
                        zip(range(-5, 5, 2), groupdf.columns)
                    ):
                        ax2.bar(
                            numofvacancy + offset,
                            groupdf.at[numofvacancy, alloy],
                            width=2,
                            color=self.alloycolor[l],
                        )
                if not os.path.exists(
                    os.path.join(self.outputdir, f"{crystal}", "SameFluc")
                ):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}", "SameFluc"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}", "SameFluc"))
                ax1.legend()
                ax1.set_xlabel("# of vacancy passed into")
                ax1.set_ylabel("Each Vacancy Average Step")
                ax2.set_ylabel("Each Group Average Step")
                plt.title(f"Steps to IDZ  under {flucname}")
                plt.tight_layout()
                plt.savefig(f"Vacancy steps to IDZ under {flucname}.jpg")
                plt.close()
                os.chdir(self.maindir)

            os.chdir(self.maindir)
        return

    def fluc_backhoppro_comparison(self):
        """Comparison under different fluc with different crystalstructure within same alloy"""
        fig, axs = plt.subplots(5, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, alloy in enumerate(self.alloysys):
                for j, (fluc, flucname) in enumerate(
                    zip(crystalflucpath, self.fluctype)
                ):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    hoppro = pd.read_json("backhopprobability.json")
                    result = hoppro.mean(axis=1)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.fluctype[j],
                        color=self.fluccolor[j],
                    )

                axs[i, typeofcrystal].set_title(alloy)
                os.chdir(self.maindir)

        fig.legend(
            self.fluctype,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameAlloy")):
            os.mkdir(os.path.join(self.outputdir, "SameAlloy"))
        os.chdir(os.path.join(self.outputdir, "SameAlloy"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "# of vacancy passed into",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Each Vacancy Backhop Probability",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Backhop probability as # of V passed ", fontsize=32)
        plt.savefig(f"Vacancy Backhop probability as fluc change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def fluc_site_comparison(self):
        """Comparison under different fluc with different crystalstructure within same alloy"""
        fig, axs = plt.subplots(5, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, alloy in enumerate(self.alloysys):
                for j, (fluc, flucname) in enumerate(
                    zip(crystalflucpath, self.fluctype)
                ):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    site = pd.read_json("site result.json")
                    result = site["Mean"]
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.fluctype[j],
                        color=self.fluccolor[j],
                    )

                os.chdir(self.maindir)
                axs[i, typeofcrystal].set_title(alloy)

        fig.legend(
            self.fluctype,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameAlloy")):
            os.mkdir(os.path.join(self.outputdir, "SameAlloy"))
        os.chdir(os.path.join(self.outputdir, "SameAlloy"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "# of vacancy passed into",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Averge Sites Visited",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Averge Sites Visited as # of V passed ", fontsize=32)
        plt.savefig(f"Averge Sites Visited as fluc change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def fluc_sd_comparison(self):
        """Comparison under different fluc with different crystalstructure within same alloy"""
        fig, axs = plt.subplots(5, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, alloy in enumerate(self.alloysys):
                for j, (fluc, flucname) in enumerate(
                    zip(crystalflucpath, self.fluctype)
                ):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    sd = pd.read_json("sd result.json")
                    result = sd["Mean"]
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.fluctype[j],
                        color=self.fluccolor[j],
                    )

                axs[i, typeofcrystal].set_title(alloy)
                os.chdir(self.maindir)
        fig.legend(
            self.fluctype,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameAlloy")):
            os.mkdir(os.path.join(self.outputdir, "SameAlloy"))
        os.chdir(os.path.join(self.outputdir, "SameAlloy"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "# of vacancy passed into",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08, 0.5, "MSD", ha="center", va="center", fontsize=24, rotation="vertical"
        )
        plt.suptitle(f"MSD as # of V passed ", fontsize=32)
        plt.savefig(f"MSD as fluc change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def fluc_step_comparison(self):
        """Comparison under different fluc with different crystalstructure within same alloy"""
        fig, axs = plt.subplots(5, 3, sharex=True, figsize=(15, 12))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, alloy in enumerate(self.alloysys):
                for j, (fluc, flucname) in enumerate(
                    zip(crystalflucpath, self.fluctype)
                ):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step = pd.read_json("steptoIDZ.json")
                    result = step.mean(axis=1)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.fluctype[j],
                        color=self.fluccolor[j],
                    )

                axs[i, typeofcrystal].set_title(alloy)
                os.chdir(self.maindir)
        fig.legend(
            self.fluctype,
            loc="upper center",
            fontsize=10,
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameAlloy")):
            os.mkdir(os.path.join(self.outputdir, "SameAlloy"))
        os.chdir(os.path.join(self.outputdir, "SameAlloy"))
        fig.text(0.24225, 0.91225, "SC", fontsize=18, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=18, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=18, ha="center", va="center")
        fig.text(
            0.5,
            0.05,
            "# of vacancy passed into",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.05,
            0.5,
            "Averge Step",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Averge Step to IDZ as # of V passed ", fontsize=28)
        plt.savefig(f"Averge Step to IDZ as fluc change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_alloy_backhoppro_comparison(self):
        fig, axs = plt.subplots(4, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, (fluc, flucname) in enumerate(zip(crystalflucpath, self.fluctype)):
                for j, alloy in enumerate(self.alloysys):

                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    hoppro = pd.read_json("backhopprobability.json")
                    result = hoppro.mean(axis=1)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.alloytype[j],
                        color=self.alloycolor[j],
                    )
                axs[i, typeofcrystal].set_title(flucname)
            os.chdir(self.maindir)

        fig.legend(
            self.alloytype,
            loc="upper center",
            fontsize=10,
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.alloytype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameFluc")):
            os.mkdir(os.path.join(self.outputdir, "SameFluc"))
        os.chdir(os.path.join(self.outputdir, "SameFluc"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "# of vacancy passed into",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Each Vacancy Backhop Probability",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Backhop probability as # of V passed ", fontsize=32)
        plt.savefig(f"Vacancy Backhop probability as alloy change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_alloy_site_comparison(self):
        fig, axs = plt.subplots(4, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, (fluc, flucname) in enumerate(zip(crystalflucpath, self.fluctype)):
                for j, alloy in enumerate(self.alloysys):

                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    site = pd.read_json("site result.json")
                    result = site["Mean"]
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.alloytype[j],
                        color=self.alloycolor[j],
                    )
                axs[i, typeofcrystal].set_title(flucname)
            os.chdir(self.maindir)

        fig.legend(
            self.alloytype,
            loc="upper center",
            fontsize=10,
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.alloytype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameFluc")):
            os.mkdir(os.path.join(self.outputdir, "SameFluc"))
        os.chdir(os.path.join(self.outputdir, "SameFluc"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "# of vacancy passed into",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Averge Sites Visited",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Averge Sites Visited as # of V passed ", fontsize=32)
        plt.savefig(f"Vacancy Backhop Averge Sites Visited as alloy change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_alloy_sd_comparison(self):
        fig, axs = plt.subplots(4, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, (fluc, flucname) in enumerate(zip(crystalflucpath, self.fluctype)):
                for j, alloy in enumerate(self.alloysys):

                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    sd = pd.read_json("sd result.json")
                    result = sd["Mean"]
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.alloytype[j],
                        color=self.alloycolor[j],
                    )
                axs[i, typeofcrystal].set_title(flucname)
            os.chdir(self.maindir)

        fig.legend(
            self.alloytype,
            loc="upper center",
            fontsize=10,
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.alloytype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameFluc")):
            os.mkdir(os.path.join(self.outputdir, "SameFluc"))
        os.chdir(os.path.join(self.outputdir, "SameFluc"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "# of vacancy passed into",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "MSD",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"MSD as # of V passed ", fontsize=32)
        plt.savefig(f" MSD as alloy change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_alloy_step_comparison(self):
        fig, axs = plt.subplots(4, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, (fluc, flucname) in enumerate(zip(crystalflucpath, self.fluctype)):
                for j, alloy in enumerate(self.alloysys):

                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step = pd.read_json("steptoIDZ.json")
                    result = step.mean(axis=1)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.alloytype[j],
                        color=self.alloycolor[j],
                    )
                axs[i, typeofcrystal].set_title(flucname)
            os.chdir(self.maindir)

        fig.legend(
            self.alloytype,
            loc="upper center",
            fontsize=10,
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.alloytype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameFluc")):
            os.mkdir(os.path.join(self.outputdir, "SameFluc"))
        os.chdir(os.path.join(self.outputdir, "SameFluc"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "# of vacancy passed into",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Average Step",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Average Step to IDZ as # of V passed ", fontsize=32)
        plt.savefig(f" Average Step to IDZ as alloy change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def analysis_fluc_concentration_profile(self):
        """
        Analysis of same fluc wih different alloy
        """
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            fig, axs = plt.subplots(
                len(self.alloytype),
                len(self.fluctype),
                sharex="col",
                figsize=(19.2, 10.8),
            )
            plot_detail = {
                "linestyle": {
                    0: "-",
                    1: (0, (1, 10)),
                    2: ":",
                    3: (0, (5, 1)),
                    4: "--",
                    5: "-.",
                },
                "marker": {
                    0: "o",
                    1: "s",
                    2: "v",
                    3: "p",
                    4: "D",
                    5: "*",
                },
                "color": {
                    "A1A2": ["black", "red"],
                    "AB1AB2": ["black", "red"],
                    "ABC1ABC2": ["black", "green", "red"],
                    "ABCD1ABCD2": ["black", "green", "yellow", "red"],
                    "ABCDE1ABCDE2": ["black", "blue", "green", "yellow", "red"],
                },
            }
            for i, (fluctype, flucpath) in enumerate(
                zip(self.fluctype, self.flucsystem)
            ):
                for j, alloy in enumerate(self.alloysys):
                    for k, numvacancy in enumerate([0, 10, 30, 50, 80, 100]):
                        os.chdir(
                            os.path.join(
                                serial_file_path,
                                f"{crystal}_{flucpath}",
                                alloy,
                                "concentrationresult",
                            )
                        )
                        concentration_df = pd.read_excel(
                            f"Concentration as position of {numvacancy}.xlsx"
                        )
                        for m, element in enumerate(self.element_type):
                            if element in concentration_df.columns:
                                axs[j, i].plot(
                                    concentration_df.index[100 - 12 : 100 + 12],
                                    concentration_df[element][100 - 12 : 100 + 12],
                                    color=plot_detail["color"][alloy][m],
                                    # linestyle=plot_detail["linestyle"][i],
                                    marker=plot_detail["marker"][k],
                                    label=element,
                                    alpha=1 - 0.15 * k,
                                )
                            os.chdir(self.maindir)

                axs[0, i].set_title(fluctype)
            os.chdir(self.outputdir)
            if not os.path.exists("continuous_concentraion_comparison"):
                os.makedirs("continuous_concentraion_comparison")
            os.chdir(os.path.join(self.outputdir, "continuous_concentraion_comparison"))
            fig.text(0.5, 0.01, "Position", ha="center")
            fig.text(0.01, 0.5, "Concentration", va="center", rotation="vertical")
            # plt.tight_layout()
            plt.subplots_adjust(
                left=0.045,
                bottom=0.052,
                right=0.983,
                top=0.975,
                wspace=0.221,
                hspace=0.052,
            )
            plt.savefig(f"{crystal}_FLUC_Continuous Concentration Profile.png")
            plt.close()
            os.chdir(self.maindir)
        return

    def analysis_alloy_concentration_profile(self):
        """
        Analysis of same alloy wih different fluctype
        """
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            fig, axs = plt.subplots(
                len(self.fluctype),
                len(self.alloytype),
                sharex="col",
                figsize=(19.2, 10.8),
            )
            plot_detail = {
                "linestyle": {
                    0: "-",
                    1: (0, (1, 10)),
                    2: ":",
                    3: (0, (5, 1)),
                    4: "--",
                    5: "-.",
                },
                "marker": {
                    0: "o",
                    1: "s",
                    2: "v",
                    3: "p",
                    4: "D",
                    5: "*",
                },
                "color": {
                    "A1A2": ["black", "red"],
                    "AB1AB2": ["black", "red"],
                    "ABC1ABC2": ["black", "green", "red"],
                    "ABCD1ABCD2": ["black", "green", "yellow", "red"],
                    "ABCDE1ABCDE2": ["black", "blue", "green", "yellow", "red"],
                },
            }
            for i, (alloytype, alloysys) in enumerate(
                zip(self.alloytype, self.alloysys)
            ):
                for j, (fluctype, flucpath) in enumerate(
                    zip(self.fluctype, self.flucsystem)
                ):
                    for k, numvacancy in enumerate([0, 10, 30, 50, 80, 100]):
                        os.chdir(
                            os.path.join(
                                serial_file_path,
                                f"{crystal}_{flucpath}",
                                alloysys,
                                "concentrationresult",
                            )
                        )
                        concentration_df = pd.read_excel(
                            f"Concentration as position of {numvacancy}.xlsx"
                        )
                        for m, element in enumerate(self.element_type):
                            if element in concentration_df.columns:
                                axs[j, i].plot(
                                    concentration_df.index[100 - 12 : 100 + 12],
                                    concentration_df[element][100 - 12 : 100 + 12],
                                    color=plot_detail["color"][alloysys][m],
                                    # linestyle=plot_detail["linestyle"][i],
                                    marker=plot_detail["marker"][k],
                                    label=element,
                                    alpha=1 - 0.15 * k,
                                )
                            os.chdir(self.maindir)

                axs[0, i].set_title(alloytype)
            os.chdir(self.outputdir)
            if not os.path.exists("continuous_concentraion_comparison"):
                os.makedirs("continuous_concentraion_comparison")
            os.chdir(os.path.join(self.outputdir, "continuous_concentraion_comparison"))
            fig.text(0.5, 0.01, "Position", ha="center")
            fig.text(0.01, 0.5, "Concentration", va="center", rotation="vertical")
            fig.text(0.575, 0.985, "nb2.5_5fluc", fontsize=12, ha="center", va="center")
            fig.text(0.575, 0.7475, "wb5_5fluc", fontsize=12, ha="center", va="center")
            fig.text(
                0.575, 0.5125, "nb2.5_10fluc", fontsize=12, ha="center", va="center"
            )
            fig.text(
                0.575, 0.27375, "wb5_10fluc", fontsize=12, ha="center", va="center"
            )
            # plt.tight_layout()
            plt.subplots_adjust(
                left=0.045,
                bottom=0.052,
                right=0.983,
                top=0.975,
                wspace=0.221,
                hspace=0.16,
            )

            plt.savefig(f"{crystal}_alloy_Continuous Concentration Profile.png")
            plt.close()
            os.chdir(self.maindir)
        return

    def draw_label(self):
        for k, vgroup in enumerate(self.vacancygroup):
            plt.scatter(
                [],
                [],
                color="black",
                marker=self.groupmarker[k],
                label=vgroup,
            )
        os.chdir(self.outputdir)
        plt.axis("off")
        plt.legend(bbox_to_anchor=(0.8, 0.8), ncol=1)
        plt.tight_layout()
        plt.savefig(f"Vacancy label.png")
        plt.close()
        os.chdir(self.maindir)
        return

    def run_alloy_comparision(self):
        print("Running of backhop pro ...")
        self.alloysystem_backhoppro_comparison()
        print("Running of site ...")
        self.alloysystem_site_comparison()
        print("Running of sd ...")
        self.alloysystem_sd_comparison()
        print("Running of step ...")
        self.alloysystem_step_comparison()
        return

    def run_fluc_comparision(self):
        print("Running of backhop pro ...")
        self.fluc_backhoppro_comparison()
        print("Running of site ...")
        self.fluc_site_comparison()
        print("Running of sd ...")
        self.fluc_sd_comparison()
        print("Running of step ...")
        self.fluc_step_comparison()
        return

    def run_crystal_alloy_comparision(self):
        print("Running of backhop pro ...")
        self.crystal_alloy_backhoppro_comparison()
        print("Running of site ...")
        self.crystal_alloy_site_comparison()
        print("Running of sd ...")
        self.crystal_alloy_sd_comparison()
        print("Running of step ...")
        self.crystal_alloy_step_comparison()
        return

    def run_analysis(self):
        self.fluc_idz_range()
        self.alloysystem_idz_range()
        self.run_alloy_comparision()
        self.run_fluc_comparision()
        self.run_crystal_alloy_comparision()
        self.analysis_alloy_concentration_profile()
        self.analysis_fluc_concentration_profile()
        return

class ErfAnalyzer:
    def __init__(self, filepath):
        self.crystalstructure = ["sc", "bcc", "fcc"]
        self.fliepath = filepath
        self.flucsystem = [
            "distribution_diffpro_data_size10_numofvacancy100_nb2.5_5fluc",
            "distribution_diffpro_data_size10_numofvacancy100_wb5_5fluc",
            "distribution_diffpro_data_size10_numofvacancy100_nb2.5_10fluc",
            "distribution_diffpro_data_size10_numofvacancy100_wb5_10fluc",
        ]

        self.fluctype = [
            "nb2.5_5fluc",
            "wb5_5fluc",
            "nb2.5_10fluc",
            "wb5_10fluc",
        ]
        self.alloytype = ["ABp", "AB", "ABC", "ABCD", "ABCDE"]
        self.alloycolor = ["black", "blue", "green", "orange", "red"]
        self.fluccolor = ["black", "blue", "red", "green"]
        self.element_type = ["A", "B", "C", "D", "E"]
        self.step = [10, 30, 50]
        self.groupmarker = ["o", "s", "v", "p", "D", "*"]
        self.alloysys = ["A1A2", "AB1AB2", "ABC1ABC2", "ABCD1ABCD2", "ABCDE1ABCDE2"]
        self.maindir = filepath
        if not os.path.exists(os.path.join(self.maindir, "ComparisonResult")):
            os.mkdir(os.path.join(self.maindir, "ComparisonResult"))
        for crystal in self.crystalstructure:
            if not os.path.exists(
                os.path.join(self.maindir, "ComparisonResult", f"{crystal}")
            ):
                os.mkdir(os.path.join(self.maindir, "ComparisonResult", f"{crystal}"))
        self.outputdir = os.path.join(self.maindir, "ComparisonResult")
        return

    def alloysystem_backhoppro_comparison(self):
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                alloysysrecord = pd.DataFrame()
                for i, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_pro = pd.read_json(f"stepbackhoppro.json")
                    # 計算每步均值
                    alloysysrecord[alloy] = step_pro.mean(axis=0)

                if not os.path.exists(
                    os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                ):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                # 繪製散佈圖
                for i, alloy in enumerate(alloysysrecord.columns):
                    plt.scatter(
                        0,
                        alloysysrecord.at[self.step[0], alloy],
                        color=self.alloycolor[i],
                        label=alloy,
                    )
                    for j, step in enumerate(self.step[1:], 1):
                        plt.scatter(
                            j, alloysysrecord.at[step, alloy], color=self.alloycolor[i]
                        )
                plt.xticks(np.arange(0, 5), [f"{step}" for step in self.step])
                plt.xlabel(f"After steps evolution")
                plt.ylabel("backhop probability")
                plt.legend()
                plt.title(
                    f"Vacancy backhop probability under {flucname} with step evolution"
                )
                plt.tight_layout()
                plt.savefig(
                    f"Vacancy backhop probability under {flucname} with step evolution.jpg"
                )
                plt.close()
                os.chdir(self.maindir)

            os.chdir(self.maindir)
        return

    def alloysystem_site_comparison(self):
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                alloysysrecord = pd.DataFrame()
                for i, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsiteresult.json")
                    # 計算每步均值
                    alloysysrecord[alloy] = step_result.mean(axis=0)

                if not os.path.exists(
                    os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                ):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                # 繪製散佈圖
                for i, alloy in enumerate(alloysysrecord.columns):
                    plt.scatter(
                        0,
                        alloysysrecord.at[self.step[0], alloy],
                        color=self.alloycolor[i],
                        label=alloy,
                    )
                    for j, step in enumerate(self.step[1:], 1):
                        plt.scatter(
                            j, alloysysrecord.at[step, alloy], color=self.alloycolor[i]
                        )
                plt.xticks(np.arange(0, 3), [f"{step}" for step in self.step])
                plt.xlabel(f"After steps evolution")
                plt.ylabel("# of sites")
                plt.legend()
                plt.title(f"Site visited under {flucname} with step evolution")
                plt.tight_layout()
                plt.savefig(f"Site visited under {flucname} with step evolution.jpg")
                plt.close()
                os.chdir(self.maindir)

            os.chdir(self.maindir)
        return

    def alloysystem_sd_comparison(self):
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                alloysysrecord = pd.DataFrame()
                for i, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsdresult.json")
                    # 計算每步均值
                    alloysysrecord[alloy] = step_result.mean(axis=0)

                if not os.path.exists(
                    os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                ):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                # 繪製散佈圖
                for i, alloy in enumerate(alloysysrecord.columns):
                    plt.scatter(
                        0,
                        alloysysrecord.at[self.step[0], alloy],
                        color=self.alloycolor[i],
                        label=alloy,
                    )
                    for j, step in enumerate(self.step[1:], 1):
                        plt.scatter(
                            j, alloysysrecord.at[step, alloy], color=self.alloycolor[i]
                        )
                plt.xticks(np.arange(0, 3), [f"{step}" for step in self.step])
                plt.xlabel(f"After steps evolution")
                plt.ylabel("MSD")
                plt.legend()
                plt.title(f"MSD under {flucname} with step evolution")
                plt.tight_layout()
                plt.savefig(f"MSD under {flucname} with step evolution.jpg")
                plt.close()
                os.chdir(self.maindir)

            os.chdir(self.maindir)
        return

    def alloysystem_position_distribution(self):
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                for step in self.step:
                    record = pd.DataFrame()
                    for alloytype, alloy in zip(self.alloytype, self.alloysys):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )
                        record[alloy] = alloy_distribution[step]
                    plt.hist(
                        record,
                        bins=np.arange(0, 20),
                        histtype="bar",
                        density=True,
                        label=record.columns,
                    )
                    if not os.path.exists(
                        os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                    ):
                        os.mkdir(
                            os.path.join(
                                self.outputdir,
                                f"{crystal}",
                                f"{flucname}",
                            )
                        )
                    os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                    plt.xlabel("Position")
                    plt.xticks(np.arange(0, 20))
                    plt.ylabel("Normalized Counts")
                    plt.legend()
                    plt.title(f"Postion Distribution after {step} steps")
                    plt.savefig(f"Position Distribution at {step}.png")
                    plt.close()
                    os.chdir(self.maindir)
            os.chdir(self.maindir)
        return

    def alloysystem_position_pdf_distribution(self):
        position = np.arange(0, 21)
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                for step in self.step:
                    record = pd.DataFrame()
                    for alloytype, alloy in zip(self.alloytype, self.alloysys):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]

                        record[alloy] = countlist / sum(countlist)

                    record.plot.bar()
                    if not os.path.exists(
                        os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                    ):
                        os.mkdir(
                            os.path.join(
                                self.outputdir,
                                f"{crystal}",
                                f"{flucname}",
                            )
                        )
                    os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                    plt.legend()
                    plt.xlabel("Position")
                    plt.ylabel("Counts")
                    plt.title(f"PDF of Postion Distribution after {step} steps")
                    plt.savefig(f"bar PDF Position Distribution at {step}.png")
                    plt.close()
                    os.chdir(self.maindir)
            os.chdir(self.maindir)
        return

    def alloysystem_position_cdf_distribution(self):
        position = np.arange(0, 21)
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                for step in self.step:
                    record = pd.DataFrame()
                    for alloytype, alloy in zip(self.alloytype, self.alloysys):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        record[alloy] = countssum

                    record.plot.bar()
                    if not os.path.exists(
                        os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                    ):
                        os.mkdir(
                            os.path.join(
                                self.outputdir,
                                f"{crystal}",
                                f"{flucname}",
                            )
                        )
                    os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                    plt.legend()
                    plt.xlabel("Position")
                    plt.ylabel("Counts")
                    plt.title(f"CDF of Postion Distribution after {step} steps")
                    plt.savefig(f"bar CDF Position Distribution at {step}.png")
                    plt.close()
                    os.chdir(self.maindir)
            os.chdir(self.maindir)
        return

    def alloy_alloysysyem_step_position_distribution(self):
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.fluctype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, step in enumerate(self.step):
                for j, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                    step_record = pd.DataFrame()
                    for k, (alloyname, alloy) in enumerate(
                        zip(self.alloytype, self.alloysys)
                    ):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )
                        step_record[alloyname] = alloy_distribution[step]
                    axs[i, j].hist(
                        step_record,
                        bins=np.arange(0, 20),
                        histtype="bar",
                        density=True,
                        label=step_record.columns,
                    )
                    if i == 0:
                        axs[0, j].set_title(flucname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.alloytype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.alloytype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(f"Position Distribution as step evolution ", fontsize=32)
            plt.savefig(f"Position Distribution as fluc change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def alloy_alloysysyem_step_position_pdf_distribution(self):
        position = np.arange(0, 21)
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.fluctype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for steppos, step in enumerate(self.step):
                for j, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                    step_record = pd.DataFrame()
                    for k, (alloyname, alloy) in enumerate(
                        zip(self.alloytype, self.alloysys)
                    ):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]

                        step_record[alloyname] = countlist / sum(countlist)
                    step_record.plot.bar(ax=axs[steppos, j], legend=False)
                    if steppos == 0:
                        axs[0, j].set_title(flucname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.alloytype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.alloytype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(
                f"PDF of Position Distribution as step evolution ", fontsize=32
            )
            plt.savefig(f"PDF of Position Distribution as fluc change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def alloy_alloysysyem_step_position_cdf_distribution(self):
        position = np.arange(0, 21)
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.fluctype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for steppos, step in enumerate(self.step):
                for j, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                    step_record = pd.DataFrame(index=np.arange(0, 10))
                    for k, (alloyname, alloy) in enumerate(
                        zip(self.alloytype, self.alloysys)
                    ):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        step_record[alloy] = countssum
                    step_record.plot.bar(ax=axs[steppos, j], legend=False)
                    if steppos == 0:
                        axs[0, j].set_title(flucname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.alloytype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.alloytype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(
                f"CDF of Position Distribution as step evolution ", fontsize=32
            )
            plt.savefig(f"CDF of Position Distribution as fluc change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def fluc_alloysysyem_step_position_distribution(self):
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.alloytype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, step in enumerate(self.step):
                for j, (alloyname, alloy) in enumerate(
                    zip(self.alloytype, self.alloysys)
                ):
                    step_record = pd.DataFrame()
                    for k, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )
                        step_record[flucname] = alloy_distribution[step]
                    axs[i, j].hist(
                        step_record,
                        bins=np.arange(0, 20),
                        histtype="bar",
                        density=True,
                        label=step_record.columns,
                    )
                    if i == 0:
                        axs[0, j].set_title(alloyname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.fluctype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.fluctype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(f"Position Distribution as step evolution ", fontsize=32)
            plt.savefig(f"Position Distribution as alloy change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def fluc_alloysysyem_step_position_pdf_distribution(self):
        position = np.arange(0, 21)
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.alloytype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for steppos, step in enumerate(self.step):
                for j, (alloyname, alloy) in enumerate(
                    zip(self.alloytype, self.alloysys)
                ):
                    step_record = pd.DataFrame(index=np.arange(0, 10))
                    for k, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]

                        step_record[flucname] = countlist / sum(countlist)
                    step_record.plot.bar(ax=axs[steppos, j], legend=False)
                    if steppos == 0:
                        axs[0, j].set_title(alloyname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.fluctype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.fluctype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(
                f"PDF of Position Distribution as step evolution ", fontsize=32
            )
            plt.savefig(f"PDF Position Distribution as alloy change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def fluc_alloysysyem_step_position_cdf_distribution(self):
        position = np.arange(0, 21)
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.alloytype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for steppos, step in enumerate(self.step):
                for j, (alloyname, alloy) in enumerate(
                    zip(self.alloytype, self.alloysys)
                ):
                    step_record = pd.DataFrame(index=np.arange(0, 10))
                    for k, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        step_record[flucname] = countssum
                    step_record.plot.bar(ax=axs[steppos, j], legend=False)
                    if steppos == 0:
                        axs[0, j].set_title(alloyname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.fluctype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.fluctype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(
                f"CDF of Position Distribution as step evolution ", fontsize=32
            )
            plt.savefig(f"CDF Position Distribution as alloy change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def crystal_fluc_backhoppro_comparison(self):
        """Comparison under different fluc with different crystalstructure within same alloy"""
        fig, axs = plt.subplots(5, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, alloy in enumerate(self.alloysys):
                for j, (fluc, flucname) in enumerate(
                    zip(crystalflucpath, self.fluctype)
                ):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepbackhoppro.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.fluctype[j],
                        color=self.fluccolor[j],
                    )

                axs[i, typeofcrystal].set_title(alloy)
                os.chdir(self.maindir)

        fig.legend(
            self.fluctype,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameAlloy")):
            os.mkdir(os.path.join(self.outputdir, "SameAlloy"))
        os.chdir(os.path.join(self.outputdir, "SameAlloy"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "step",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Vacancy Backhop Probability",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Backhop probability as step evolution ", fontsize=32)
        plt.savefig(f"Vacancy Backhop probability as fluc change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_fluc_site_comparison(self):
        """Comparison under different fluc with different crystalstructure within same alloy"""
        fig, axs = plt.subplots(5, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, alloy in enumerate(self.alloysys):
                for j, (fluc, flucname) in enumerate(
                    zip(crystalflucpath, self.fluctype)
                ):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsiteresult.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.fluctype[j],
                        color=self.fluccolor[j],
                    )

                os.chdir(self.maindir)
                axs[i, typeofcrystal].set_title(alloy)

        fig.legend(
            self.fluctype,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameAlloy")):
            os.mkdir(os.path.join(self.outputdir, "SameAlloy"))
        os.chdir(os.path.join(self.outputdir, "SameAlloy"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "step",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Averge Sites Visited",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Averge Sites Visited as step", fontsize=32)
        plt.savefig(f"Averge Sites Visited as fluc change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_fluc_sd_comparison(self):
        """Comparison under different fluc with different crystalstructure within same alloy"""
        fig, axs = plt.subplots(5, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, alloy in enumerate(self.alloysys):
                for j, (fluc, flucname) in enumerate(
                    zip(crystalflucpath, self.fluctype)
                ):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsdresult.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.fluctype[j],
                        color=self.fluccolor[j],
                    )

                axs[i, typeofcrystal].set_title(alloy)
                os.chdir(self.maindir)
        fig.legend(
            self.fluctype,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameAlloy")):
            os.mkdir(os.path.join(self.outputdir, "SameAlloy"))
        os.chdir(os.path.join(self.outputdir, "SameAlloy"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "step",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08, 0.5, "MSD", ha="center", va="center", fontsize=24, rotation="vertical"
        )
        plt.suptitle(f"MSD as step ", fontsize=32)
        plt.savefig(f"MSD as fluc change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_alloy_backhoppro_comparison(self):
        fig, axs = plt.subplots(4, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, (fluc, flucname) in enumerate(zip(crystalflucpath, self.fluctype)):
                for j, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepbackhoppro.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.alloytype[j],
                        color=self.alloycolor[j],
                    )
                axs[i, typeofcrystal].set_title(flucname)
            os.chdir(self.maindir)

        fig.legend(
            self.alloytype,
            loc="upper center",
            fontsize=10,
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameFluc")):
            os.mkdir(os.path.join(self.outputdir, "SameFluc"))
        os.chdir(os.path.join(self.outputdir, "SameFluc"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "step",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Vacancy Backhop Probability",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Backhop probability as step evolve", fontsize=32)
        plt.savefig(f"Vacancy Backhop probability as alloy change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_alloy_site_comparison(self):
        fig, axs = plt.subplots(4, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, (fluc, flucname) in enumerate(zip(crystalflucpath, self.fluctype)):
                for j, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsiteresult.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.alloytype[j],
                        color=self.alloycolor[j],
                    )
                axs[i, typeofcrystal].set_title(flucname)
            os.chdir(self.maindir)

        fig.legend(
            self.alloytype,
            loc="upper center",
            fontsize=10,
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameFluc")):
            os.mkdir(os.path.join(self.outputdir, "SameFluc"))
        os.chdir(os.path.join(self.outputdir, "SameFluc"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "step",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Averge Sites Visited",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Averge Sites Visited as # of V passed ", fontsize=32)
        plt.savefig(f"Vacancy Backhop Averge Sites Visited as alloy change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_alloy_sd_comparison(self):
        fig, axs = plt.subplots(4, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, (fluc, flucname) in enumerate(zip(crystalflucpath, self.fluctype)):
                for j, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsdresult.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.alloytype[j],
                        color=self.alloycolor[j],
                    )
                axs[i, typeofcrystal].set_title(flucname)
            os.chdir(self.maindir)

        fig.legend(
            self.alloytype,
            loc="upper center",
            fontsize=10,
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameFluc")):
            os.mkdir(os.path.join(self.outputdir, "SameFluc"))
        os.chdir(os.path.join(self.outputdir, "SameFluc"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "# of vacancy passed into",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "MSD",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"MSD as # of V passed ", fontsize=32)
        plt.savefig(f" MSD as alloy change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def draw_label(self):
        for k, vgroup in enumerate(self.vacancygroup):
            plt.scatter(
                [],
                [],
                color="black",
                marker=self.groupmarker[k],
                label=vgroup,
            )
        os.chdir(self.outputdir)
        plt.axis("off")
        plt.legend(bbox_to_anchor=(0.8, 0.8), ncol=1)
        plt.tight_layout()
        plt.savefig(f"Vacancy label.png")
        plt.close()
        os.chdir(self.maindir)
        return

    def run_alloy_comparision(self):
        print("Running of backhop pro ...")
        self.alloysystem_backhoppro_comparison()
        print("Running of site ...")
        self.alloysystem_site_comparison()
        print("Running of sd ...")
        self.alloysystem_sd_comparison()
        return

    def run_vacancy_position_distribution(self):
        self.alloysystem_position_distribution()
        self.alloysystem_position_pdf_distribution()
        self.alloysystem_position_cdf_distribution()
        self.alloy_alloysysyem_step_position_distribution()
        self.alloy_alloysysyem_step_position_pdf_distribution()
        self.alloy_alloysysyem_step_position_cdf_distribution()
        self.fluc_alloysysyem_step_position_distribution()
        self.fluc_alloysysyem_step_position_cdf_distribution()
        return

    def run_fluc_comparision(self):
        print("Running of backhop pro ...")
        self.crystal_fluc_backhoppro_comparison()
        print("Running of site ...")
        self.crystal_fluc_site_comparison()
        print("Running of sd ...")
        self.crystal_fluc_sd_comparison()
        return

    def run_crystal_alloy_comparision(self):
        print("Running of backhop pro ...")
        self.crystal_alloy_backhoppro_comparison()
        print("Running of site ...")
        self.crystal_alloy_site_comparison()
        print("Running of sd ...")
        self.crystal_alloy_sd_comparison()
        return

    def run_analysis(self):
        # self.run_alloy_comparision()
        self.run_fluc_comparision()
        self.run_crystal_alloy_comparision()
        # self.run_vacancy_position_distribution()
        return

class GraAnalyzer:
    def __init__(self, filepath):
        self.crystalstructure = ["sc", "bcc", "fcc"]
        self.flucsystem = [
            "gradient_diffpro_data_size10_numofvacancy100_nb2.5_5fluc",
            "gradient_diffpro_data_size10_numofvacancy100_wb5_5fluc",
            "gradient_diffpro_data_size10_numofvacancy100_nb2.5_10fluc",
            "gradient_diffpro_data_size10_numofvacancy100_wb5_10fluc",
        ]
        self.fliepath = filepath
        self.fluctype = [
            "nb2.5_5fluc",
            "wb5_5fluc",
            "nb2.5_10fluc",
            "wb5_10fluc",
        ]
        self.alloytype = ["ABp", "AB", "ABC", "ABCD", "ABCDE"]
        self.alloycolor = ["black", "blue", "green", "orange", "red"]
        self.fluccolor = ["black", "blue", "red", "green"]
        self.element_type = ["A", "B", "C", "D", "E"]
        self.step = [10, 20, 30, 40, 50]
        self.groupmarker = ["o", "s", "v", "p", "D", "*"]
        self.alloysys = ["A1A2", "AB1AB2", "ABC1ABC2", "ABCD1ABCD2", "ABCDE1ABCDE2"]
        self.maindir = filepath
        if not os.path.exists(os.path.join(self.maindir, "GraComparisonResult")):
            os.mkdir(os.path.join(self.maindir, "GraComparisonResult"))
        for crystal in self.crystalstructure:
            if not os.path.exists(
                os.path.join(self.maindir, "GraComparisonResult", f"{crystal}")
            ):
                os.mkdir(
                    os.path.join(self.maindir, "GraComparisonResult", f"{crystal}")
                )
        self.outputdir = os.path.join(self.maindir, "GraComparisonResult")
        return

    def alloysystem_backhoppro_comparison(self):
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                alloysysrecord = pd.DataFrame()
                for i, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_pro = pd.read_json(f"stepbackhoppro.json")
                    # 計算每步均值
                    alloysysrecord[alloy] = step_pro.mean(axis=0)

                if not os.path.exists(
                    os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                ):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                # 繪製散佈圖
                for i, alloy in enumerate(alloysysrecord.columns):
                    plt.scatter(
                        0,
                        alloysysrecord.at[self.step[0], alloy],
                        color=self.alloycolor[i],
                        label=alloy,
                    )
                    for j, step in enumerate(self.step[1:], 1):
                        plt.scatter(
                            j, alloysysrecord.at[step, alloy], color=self.alloycolor[i]
                        )
                plt.xticks(np.arange(0, 5), [f"{step}" for step in self.step])
                plt.xlabel(f"After steps evolution")
                plt.ylabel("backhop probability")
                plt.legend()
                plt.title(
                    f"Vacancy backhop probability under {flucname} with step evolution"
                )
                plt.tight_layout()
                plt.savefig(
                    f"Vacancy backhop probability under {flucname} with step evolution.jpg"
                )
                plt.close()
                os.chdir(self.maindir)

            os.chdir(self.maindir)
        return

    def alloysystem_site_comparison(self):
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                alloysysrecord = pd.DataFrame()
                for i, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsiteresult.json")
                    # 計算每步均值
                    alloysysrecord[alloy] = step_result.mean(axis=0)

                if not os.path.exists(
                    os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                ):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                # 繪製散佈圖
                for i, alloy in enumerate(alloysysrecord.columns):
                    plt.scatter(
                        0,
                        alloysysrecord.at[self.step[0], alloy],
                        color=self.alloycolor[i],
                        label=alloy,
                    )
                    for j, step in enumerate(self.step[1:], 1):
                        plt.scatter(
                            j, alloysysrecord.at[step, alloy], color=self.alloycolor[i]
                        )
                plt.xticks(np.arange(0, 5), [f"{step}" for step in self.step])
                plt.xlabel(f"After steps evolution")
                plt.ylabel("# of sites")
                plt.legend()
                plt.title(f"Site visited under {flucname} with step evolution")
                plt.tight_layout()
                plt.savefig(f"Site visited under {flucname} with step evolution.jpg")
                plt.close()
                os.chdir(self.maindir)

            os.chdir(self.maindir)
        return

    def alloysystem_sd_comparison(self):
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                alloysysrecord = pd.DataFrame()
                for i, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsdresult.json")
                    # 計算每步均值
                    alloysysrecord[alloy] = step_result.mean(axis=0)

                if not os.path.exists(
                    os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                ):
                    os.mkdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                # 繪製散佈圖
                for i, alloy in enumerate(alloysysrecord.columns):
                    plt.scatter(
                        0,
                        alloysysrecord.at[self.step[0], alloy],
                        color=self.alloycolor[i],
                        label=alloy,
                    )
                    for j, step in enumerate(self.step[1:], 1):
                        plt.scatter(
                            j, alloysysrecord.at[step, alloy], color=self.alloycolor[i]
                        )
                plt.xticks(np.arange(0, 5), [f"{step}" for step in self.step])
                plt.xlabel(f"After steps evolution")
                plt.ylabel("MSD")
                plt.legend()
                plt.title(f"MSD under {flucname} with step evolution")
                plt.tight_layout()
                plt.savefig(f"MSD under {flucname} with step evolution.jpg")
                plt.close()
                os.chdir(self.maindir)

            os.chdir(self.maindir)
        return

    def alloysystem_position_distribution(self):
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                for step in self.step:
                    record = pd.DataFrame()
                    for alloytype, alloy in zip(self.alloytype, self.alloysys):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )
                        record[alloy] = alloy_distribution[step]
                    plt.hist(
                        record,
                        bins=np.arange(0, 20),
                        histtype="bar",
                        density=True,
                        label=record.columns,
                    )
                    if not os.path.exists(
                        os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                    ):
                        os.mkdir(
                            os.path.join(
                                self.outputdir,
                                f"{crystal}",
                                f"{flucname}",
                            )
                        )
                    os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                    plt.xlabel("Position")
                    plt.xticks(np.arange(0, 20))
                    plt.ylabel("Normalized Counts")
                    plt.legend()
                    plt.title(f"Postion Distribution after {step} steps")
                    plt.savefig(f"Position Distribution at {step}.png")
                    plt.close()
                    os.chdir(self.maindir)
            os.chdir(self.maindir)
        return

    def alloysystem_position_pdf_distribution(self):
        position = np.arange(0, 21)
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                for step in self.step:
                    record = pd.DataFrame()
                    for alloytype, alloy in zip(self.alloytype, self.alloysys):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]

                        record[alloy] = countlist / sum(countlist)

                    record.plot.bar()
                    if not os.path.exists(
                        os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                    ):
                        os.mkdir(
                            os.path.join(
                                self.outputdir,
                                f"{crystal}",
                                f"{flucname}",
                            )
                        )
                    os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                    plt.legend()
                    plt.xlabel("Position")
                    plt.ylabel("Counts")
                    plt.title(f"PDF of Postion Distribution after {step} steps")
                    plt.savefig(f"bar PDF Position Distribution at {step}.png")
                    plt.close()
                    os.chdir(self.maindir)
            os.chdir(self.maindir)
        return

    def alloysystem_position_cdf_distribution(self):
        position = np.arange(0, 21)
        for crystal in self.crystalstructure:
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for fluc, flucname in zip(flucpath, self.fluctype):
                for step in self.step:
                    record = pd.DataFrame()
                    for alloytype, alloy in zip(self.alloytype, self.alloysys):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        record[alloy] = countssum

                    record.plot.bar()
                    if not os.path.exists(
                        os.path.join(self.outputdir, f"{crystal}", f"{flucname}")
                    ):
                        os.mkdir(
                            os.path.join(
                                self.outputdir,
                                f"{crystal}",
                                f"{flucname}",
                            )
                        )
                    os.chdir(os.path.join(self.outputdir, f"{crystal}", f"{flucname}"))
                    plt.legend()
                    plt.xlabel("Position")
                    plt.ylabel("Counts")
                    plt.title(f"CDF of Postion Distribution after {step} steps")
                    plt.savefig(f"bar CDF Position Distribution at {step}.png")
                    plt.close()
                    os.chdir(self.maindir)
            os.chdir(self.maindir)
        return

    def alloy_alloysysyem_step_position_distribution(self):
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.fluctype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, step in enumerate(self.step):
                for j, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                    step_record = pd.DataFrame()
                    for k, (alloyname, alloy) in enumerate(
                        zip(self.alloytype, self.alloysys)
                    ):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )
                        step_record[alloyname] = alloy_distribution[step]
                    axs[i, j].hist(
                        step_record,
                        bins=np.arange(0, 20),
                        histtype="bar",
                        density=True,
                        label=step_record.columns,
                    )
                    if i == 0:
                        axs[0, j].set_title(flucname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.alloytype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.alloytype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(f"Position Distribution as step evolution ", fontsize=32)
            plt.savefig(f"Position Distribution as fluc change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def alloy_alloysysyem_step_position_pdf_distribution(self):
        position = np.arange(0, 21)
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.fluctype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for steppos, step in enumerate(self.step):
                for j, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                    step_record = pd.DataFrame()
                    for k, (alloyname, alloy) in enumerate(
                        zip(self.alloytype, self.alloysys)
                    ):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]

                        step_record[alloyname] = countlist / sum(countlist)
                    step_record.plot.bar(ax=axs[steppos, j], legend=False)
                    if steppos == 0:
                        axs[0, j].set_title(flucname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.alloytype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.alloytype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(
                f"PDF of Position Distribution as step evolution ", fontsize=32
            )
            plt.savefig(f"PDF of Position Distribution as fluc change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def alloy_alloysysyem_step_position_cdf_distribution(self):
        position = np.arange(0, 21)
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.fluctype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for steppos, step in enumerate(self.step):
                for j, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                    step_record = pd.DataFrame(index=np.arange(0, 10))
                    for k, (alloyname, alloy) in enumerate(
                        zip(self.alloytype, self.alloysys)
                    ):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        step_record[alloy] = countssum
                    step_record.plot.bar(ax=axs[steppos, j], legend=False)
                    if steppos == 0:
                        axs[0, j].set_title(flucname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.alloytype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.alloytype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(
                f"CDF of Position Distribution as step evolution ", fontsize=32
            )
            plt.savefig(f"CDF of Position Distribution as fluc change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def fluc_alloysysyem_step_position_distribution(self):
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.alloytype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, step in enumerate(self.step):
                for j, (alloyname, alloy) in enumerate(
                    zip(self.alloytype, self.alloysys)
                ):
                    step_record = pd.DataFrame()
                    for k, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )
                        step_record[flucname] = alloy_distribution[step]
                    axs[i, j].hist(
                        step_record,
                        bins=np.arange(0, 20),
                        histtype="bar",
                        density=True,
                        label=step_record.columns,
                    )
                    if i == 0:
                        axs[0, j].set_title(alloyname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.fluctype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.fluctype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(f"Position Distribution as step evolution ", fontsize=32)
            plt.savefig(f"Position Distribution as alloy change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def fluc_alloysysyem_step_position_pdf_distribution(self):
        position = np.arange(0, 21)
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.alloytype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for steppos, step in enumerate(self.step):
                for j, (alloyname, alloy) in enumerate(
                    zip(self.alloytype, self.alloysys)
                ):
                    step_record = pd.DataFrame(index=np.arange(0, 10))
                    for k, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]

                        step_record[flucname] = countlist / sum(countlist)
                    step_record.plot.bar(ax=axs[steppos, j], legend=False)
                    if steppos == 0:
                        axs[0, j].set_title(alloyname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.fluctype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.fluctype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(
                f"PDF of Position Distribution as step evolution ", fontsize=32
            )
            plt.savefig(f"PDF Position Distribution as alloy change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def fluc_alloysysyem_step_position_cdf_distribution(self):
        position = np.arange(0, 21)
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            fig, axs = plt.subplots(
                len(self.step), len(self.alloytype), sharex=True, figsize=(20, 15)
            )
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            flucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for steppos, step in enumerate(self.step):
                for j, (alloyname, alloy) in enumerate(
                    zip(self.alloytype, self.alloysys)
                ):
                    step_record = pd.DataFrame(index=np.arange(0, 10))
                    for k, (fluc, flucname) in enumerate(zip(flucpath, self.fluctype)):
                        os.chdir(
                            os.path.join(serial_file_path, fluc, alloy, "distribution")
                        )
                        alloy_distribution = pd.read_json(
                            f"Vacancy distribution of {alloy}.json"
                        )[step].values.flatten()
                        hist, bins = np.histogram(alloy_distribution, bins=position)
                        if len(bins) % 2 == 0:
                            countlist = np.empty(len(hist) // 2 + 1)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countlist[-1] = hist[len(hist) // 2]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        else:
                            countlist = np.empty(len(hist) // 2)
                            for i, pos in enumerate(range(len(hist) // 2)):
                                countlist[i] = hist[pos] + hist[len(hist) - 1 - i]
                            countssum = np.cumsum(countlist) / sum(countlist)
                        step_record[flucname] = countssum
                    step_record.plot.bar(ax=axs[steppos, j], legend=False)
                    if steppos == 0:
                        axs[0, j].set_title(alloyname)

            if not os.path.exists(os.path.join(self.outputdir, f"{crystal}")):
                os.mkdir(os.path.join(self.outputdir, f"{crystal}"))
            os.chdir(os.path.join(self.outputdir, f"{crystal}"))
            fig.legend(
                self.fluctype,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(self.fluctype),
            )
            fig.text(
                0.5,
                0.075,
                "Position",
                fontsize=24,
                ha="center",
                va="center",
            )
            fig.text(
                0.08,
                0.5,
                "Counts",
                ha="center",
                va="center",
                fontsize=24,
                rotation="vertical",
            )
            plt.suptitle(
                f"CDF of Position Distribution as step evolution ", fontsize=32
            )
            plt.savefig(f"CDF Position Distribution as alloy change.jpg")
            plt.close()
            os.chdir(self.maindir)
        return

    def crystal_fluc_backhoppro_comparison(self):
        """Comparison under different fluc with different crystalstructure within same alloy"""
        fig, axs = plt.subplots(5, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, alloy in enumerate(self.alloysys):
                for j, (fluc, flucname) in enumerate(
                    zip(crystalflucpath, self.fluctype)
                ):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepbackhoppro.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.fluctype[j],
                        color=self.fluccolor[j],
                    )

                axs[i, typeofcrystal].set_title(alloy)
                os.chdir(self.maindir)

        fig.legend(
            self.fluctype,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameAlloy")):
            os.mkdir(os.path.join(self.outputdir, "SameAlloy"))
        os.chdir(os.path.join(self.outputdir, "SameAlloy"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "step",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Vacancy Backhop Probability",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Backhop probability as step evolution ", fontsize=32)
        plt.savefig(f"Vacancy Backhop probability as fluc change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_fluc_site_comparison(self):
        """Comparison under different fluc with different crystalstructure within same alloy"""
        fig, axs = plt.subplots(5, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, alloy in enumerate(self.alloysys):
                for j, (fluc, flucname) in enumerate(
                    zip(crystalflucpath, self.fluctype)
                ):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsiteresult.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.fluctype[j],
                        color=self.fluccolor[j],
                    )

                os.chdir(self.maindir)
                axs[i, typeofcrystal].set_title(alloy)

        fig.legend(
            self.fluctype,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameAlloy")):
            os.mkdir(os.path.join(self.outputdir, "SameAlloy"))
        os.chdir(os.path.join(self.outputdir, "SameAlloy"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "step",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Averge Sites Visited",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Averge Sites Visited as step", fontsize=32)
        plt.savefig(f"Averge Sites Visited as fluc change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_fluc_sd_comparison(self):
        """Comparison under different fluc with different crystalstructure within same alloy"""
        fig, axs = plt.subplots(5, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, alloy in enumerate(self.alloysys):
                for j, (fluc, flucname) in enumerate(
                    zip(crystalflucpath, self.fluctype)
                ):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsdresult.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.fluctype[j],
                        color=self.fluccolor[j],
                    )

                axs[i, typeofcrystal].set_title(alloy)
                os.chdir(self.maindir)
        fig.legend(
            self.fluctype,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameAlloy")):
            os.mkdir(os.path.join(self.outputdir, "SameAlloy"))
        os.chdir(os.path.join(self.outputdir, "SameAlloy"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "step",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08, 0.5, "MSD", ha="center", va="center", fontsize=24, rotation="vertical"
        )
        plt.suptitle(f"MSD as step ", fontsize=32)
        plt.savefig(f"MSD as fluc change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_alloy_backhoppro_comparison(self):
        fig, axs = plt.subplots(4, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, (fluc, flucname) in enumerate(zip(crystalflucpath, self.fluctype)):
                for j, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepbackhoppro.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.alloytype[j],
                        color=self.alloycolor[j],
                    )
                axs[i, typeofcrystal].set_title(flucname)
            os.chdir(self.maindir)

        fig.legend(
            self.alloytype,
            loc="upper center",
            fontsize=10,
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameFluc")):
            os.mkdir(os.path.join(self.outputdir, "SameFluc"))
        os.chdir(os.path.join(self.outputdir, "SameFluc"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "step",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Vacancy Backhop Probability",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Backhop probability as step evolve", fontsize=32)
        plt.savefig(f"Vacancy Backhop probability as alloy change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_alloy_site_comparison(self):
        fig, axs = plt.subplots(4, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, (fluc, flucname) in enumerate(zip(crystalflucpath, self.fluctype)):
                for j, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsiteresult.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.alloytype[j],
                        color=self.alloycolor[j],
                    )
                axs[i, typeofcrystal].set_title(flucname)
            os.chdir(self.maindir)

        fig.legend(
            self.alloytype,
            loc="upper center",
            fontsize=10,
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameFluc")):
            os.mkdir(os.path.join(self.outputdir, "SameFluc"))
        os.chdir(os.path.join(self.outputdir, "SameFluc"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "step",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "Averge Sites Visited",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"Averge Sites Visited as # of V passed ", fontsize=32)
        plt.savefig(f"Vacancy Backhop Averge Sites Visited as alloy change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def crystal_alloy_sd_comparison(self):
        fig, axs = plt.subplots(4, 3, sharex=True, figsize=(20, 15))
        for typeofcrystal, crystal in enumerate(self.crystalstructure):
            serial_file_path = os.path.join(self.fliepath, rf"{crystal}Lattice")
            crystalflucpath = [f"{crystal}_{fluc}" for fluc in self.flucsystem]
            for i, (fluc, flucname) in enumerate(zip(crystalflucpath, self.fluctype)):
                for j, alloy in enumerate(self.alloysys):
                    os.chdir(
                        os.path.join(serial_file_path, fluc, alloy, "simulationresult")
                    )
                    step_result = pd.read_json(f"stepsdresult.json")
                    result = step_result.mean(axis=0)
                    axs[i, typeofcrystal].scatter(
                        result.index,
                        result,
                        label=self.alloytype[j],
                        color=self.alloycolor[j],
                    )
                axs[i, typeofcrystal].set_title(flucname)
            os.chdir(self.maindir)

        fig.legend(
            self.alloytype,
            loc="upper center",
            fontsize=10,
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(self.fluctype),
        )
        if not os.path.exists(os.path.join(self.outputdir, "SameFluc")):
            os.mkdir(os.path.join(self.outputdir, "SameFluc"))
        os.chdir(os.path.join(self.outputdir, "SameFluc"))
        fig.text(0.24225, 0.91225, "SC", fontsize=20, ha="center", va="center")
        fig.text(0.51425, 0.91225, "BCC", fontsize=20, ha="center", va="center")
        fig.text(0.785, 0.91225, "FCC", fontsize=20, ha="center", va="center")
        fig.text(
            0.5,
            0.075,
            "# of vacancy passed into",
            fontsize=24,
            ha="center",
            va="center",
        )
        fig.text(
            0.08,
            0.5,
            "MSD",
            ha="center",
            va="center",
            fontsize=24,
            rotation="vertical",
        )
        plt.suptitle(f"MSD as # of V passed ", fontsize=32)
        plt.savefig(f" MSD as alloy change.jpg")
        plt.close()
        os.chdir(self.maindir)
        return

    def draw_label(self):
        for k, vgroup in enumerate(self.vacancygroup):
            plt.scatter(
                [],
                [],
                color="black",
                marker=self.groupmarker[k],
                label=vgroup,
            )
        os.chdir(self.outputdir)
        plt.axis("off")
        plt.legend(bbox_to_anchor=(0.8, 0.8), ncol=1)
        plt.tight_layout()
        plt.savefig(f"Vacancy label.png")
        plt.close()
        os.chdir(self.maindir)
        return

    def run_alloy_comparision(self):
        print("Running of backhop pro ...")
        self.alloysystem_backhoppro_comparison()
        print("Running of site ...")
        self.alloysystem_site_comparison()
        print("Running of sd ...")
        self.alloysystem_sd_comparison()
        return

    def run_vacancy_position_distribution(self):
        self.alloysystem_position_distribution()
        self.alloysystem_position_pdf_distribution()
        self.alloysystem_position_cdf_distribution()
        self.alloy_alloysysyem_step_position_distribution()
        self.alloy_alloysysyem_step_position_pdf_distribution()
        self.alloy_alloysysyem_step_position_cdf_distribution()
        self.fluc_alloysysyem_step_position_distribution()
        self.fluc_alloysysyem_step_position_cdf_distribution()
        return

    def run_fluc_comparision(self):
        print("Running of backhop pro ...")
        self.crystal_fluc_backhoppro_comparison()
        print("Running of site ...")
        self.crystal_fluc_site_comparison()
        print("Running of sd ...")
        self.crystal_fluc_sd_comparison()
        return

    def run_crystal_alloy_comparision(self):
        print("Running of backhop pro ...")
        self.crystal_alloy_backhoppro_comparison()
        print("Running of site ...")
        self.crystal_alloy_site_comparison()
        print("Running of sd ...")
        self.crystal_alloy_sd_comparison()
        return

    def run_analysis(self):
        self.run_alloy_comparision()
        self.run_fluc_comparision()
        self.run_crystal_alloy_comparision()
        self.run_vacancy_position_distribution()
        return


def analyze_serial_alloy(alloy):
    main_dir = os.path.abspath(os.curdir)
    os.chdir(os.path.join(main_dir, alloy))
    print(f"Now in {alloy}.")
    parameter_dict = {
        "alloy": None,
        "repetitions": None,
        "ratio": None,
        "numofvacancy": None,
        "lattice_size": None,
        "steps": None,
    }
    with open(f"{alloy}_Parameter.txt", "r") as fh:
        for para in fh:
            key, value = para.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in parameter_dict:
                parameter_dict[key] = value
    lattice_record = pd.read_json(
        f"Lattice record of {parameter_dict['alloy']} with {int(parameter_dict['numofvacancy'])} json.gz",
        compression="gzip",
    )
    v_position_record = pd.read_json(
        f"Vacancy position of {parameter_dict['alloy']} with {int(parameter_dict['numofvacancy'])} json.gz",
        compression="gzip",
    )
    vacancy_record = pd.read_json(
        f"Vacancy trace of {parameter_dict['alloy']} with {int(parameter_dict['numofvacancy'])} json.gz",
        compression="gzip",
    )
    analyzer = SerialInterdiffusionAnalyzer(
        parameter_dict["alloy"],
        int(parameter_dict["repetitions"]),
        int(parameter_dict["lattice_size"]),
        float(ast.literal_eval(parameter_dict["ratio"])[1]),
        float(ast.literal_eval(parameter_dict["ratio"])[0]),
        lattice_record,
        v_position_record,
        vacancy_record,
    )
    analyzer.run_concentration_analysis()
    analyzer.run_vacancy_behavior_analysis()
    os.chdir(main_dir)
    return


def analyze_distribution_alloy(alloy):
    main_dir = os.path.abspath(os.curdir)
    os.chdir(os.path.join(main_dir, alloy))
    print(f"Now in {alloy}.")
    parameter_dict = {
        "alloy": None,
        "repetitions": None,
        "ratio": None,
        "numofvacancy": None,
        "lattice_size": None,
        "steps": None,
    }
    with open(f"{alloy}_Parameter.txt", "r") as fh:
        for para in fh:
            key, value = para.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in parameter_dict:
                parameter_dict[key] = value
    lattice_record = pd.read_json(
        f"Lattice record of {parameter_dict['alloy']} with {int(parameter_dict['numofvacancy'])} json.gz",
        compression="gzip",
    )
    v_position_record = pd.read_json(
        f"Vacancy position of {parameter_dict['alloy']} with {int(parameter_dict['numofvacancy'])} json.gz",
        compression="gzip",
    )
    vacancy_record = pd.read_json(
        f"Vacancy trace of {parameter_dict['alloy']} with {int(parameter_dict['numofvacancy'])} json.gz",
        compression="gzip",
    )
    analyzer = ErfInterdiffusionAnalyzer(
        parameter_dict["alloy"],
        int(parameter_dict["repetitions"]),
        int(parameter_dict["lattice_size"]),
        float(ast.literal_eval(parameter_dict["ratio"])[1]),
        float(ast.literal_eval(parameter_dict["ratio"])[0]),
        lattice_record,
        v_position_record,
        vacancy_record,
    )
    analyzer.run_concentration_analysis()
    analyzer.run_vacancy_behavior_analysis()
    os.chdir(main_dir)
    return


def analyze_gradient_alloy(alloy):
    main_dir = os.path.abspath(os.curdir)
    os.chdir(os.path.join(main_dir, alloy))
    print(f"Now in {alloy}.")
    parameter_dict = {
        "alloy": None,
        "repetitions": None,
        "ratio": None,
        "numofvacancy": None,
        "lattice_size": None,
        "steps": None,
    }
    with open(f"{alloy}_Parameter.txt", "r") as fh:
        for para in fh:
            key, value = para.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in parameter_dict:
                parameter_dict[key] = value
    lattice_record = pd.read_json(
        f"Lattice record of {parameter_dict['alloy']} with {int(parameter_dict['numofvacancy'])} json.gz",
        compression="gzip",
    )
    v_position_record = pd.read_json(
        f"Vacancy position of {parameter_dict['alloy']} with {int(parameter_dict['numofvacancy'])} json.gz",
        compression="gzip",
    )
    vacancy_record = pd.read_json(
        f"Vacancy trace of {parameter_dict['alloy']} with {int(parameter_dict['numofvacancy'])} json.gz",
        compression="gzip",
    )
    analyzer = GradientInterdiffusionAnalyzer(
        parameter_dict["alloy"],
        int(parameter_dict["repetitions"]),
        int(parameter_dict["lattice_size"]),
        float(ast.literal_eval(parameter_dict["ratio"])[1]),
        float(ast.literal_eval(parameter_dict["ratio"])[0]),
        lattice_record,
        v_position_record,
        vacancy_record,
    )
    analyzer.run_concentration_analysis()
    analyzer.run_vacancy_behavior_analysis()
    os.chdir(main_dir)
    return
