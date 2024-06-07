import os, time, glob, ast
import numpy as np
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import multiprocessing as mp
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression


class VacancyAnalyzer:
    def __init__(self, crystalstructure, KMCsteps, alloy_system, vacancy_record):
        self.alloy_system = alloy_system
        self.steps = KMCsteps
        self.vacancy_record = vacancy_record
        self.x_columns = [col for col in self.vacancy_record.columns if "x" in col]
        self.y_columns = [col for col in self.vacancy_record.columns if "y" in col]
        self.z_columns = [col for col in self.vacancy_record.columns if "z" in col]
        self.sd_columns = [col for col in self.vacancy_record.columns if "SD" in col]
        self.time_columns = [
            col for col in self.vacancy_record.columns if "Time" in col
        ]
        self.trial_times = len(self.x_columns)
        self.crystalstructure = crystalstructure
        self.maindir = os.path.abspath(os.curdir)
        self.spacing = 2 
    # Trace Visualization
    def vacancy_trace_counts_contourf_plotting(self, trial, count_table, x, y, cmap):
        count_df = count_table.pivot_table(
            index=y, columns=x, values="count", fill_value=0
        )
        x_values = count_df.columns
        y_values = count_df.index

        X_coor, Y_coor = np.meshgrid(x_values, y_values)
        count_value = count_df.values
        levels = np.arange(count_value.min(), count_value.max() + 1)
        norm_contour = Normalize(vmin=1, vmax=50)
        plt.contour(
            X_coor, Y_coor, count_value, levels=levels, cmap=cmap, norm=norm_contour
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim([-300, 300])
        plt.ylim([-300, 300])
        plt.title("Vacancy Trace Contour distribution counts")
        plt.savefig(
            f"contourresult/{trial} Vacany trace  Contour distribution counts of {self.alloy_system}.png"
        )
        plt.close()
        return

    def vacancy_trace_counts_plotting(self):
        """
        Calculate each simulation of how randomly vacancy walk by counting # of position it visited
        """
        # plot trace counts
        norm = Normalize(vmin=1, vmax=50)
        for trial, (trial_x, trial_y, trial_z) in enumerate(
            zip(self.x_columns, self.y_columns, self.z_columns)
        ):
            df = self.vacancy_record[[trial_x, trial_y, trial_z]].copy()
            df["count"] = 1
            site_counts = (
                df.groupby([trial_x, trial_y, trial_z])
                .agg({"count": len})
                .reset_index()
            )
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            cmap = mpl.colormaps["jet"]
            cmap.set_under(color="white")

            scatter = ax.scatter(
                site_counts[trial_x],
                site_counts[trial_y],
                site_counts[trial_z],
                c=site_counts["count"],
                cmap=cmap,
                norm=norm,
            )

            ax.set_xlim([-300, 300])
            ax.set_ylim([-300, 300])
            ax.set_zlim([-300, 300])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Count")

            plt.title("Vacancy Trace distribution counts")
            plt.tight_layout()
            plt.savefig(
                f"countplottingresult/{trial} Vacany trace distribution counts of {self.alloy_system}.png"
            )
            plt.close()

            projection_site_counts = (
                df.groupby([trial_x, trial_y]).agg({"count": len}).reset_index()
            )
            self.vacancy_trace_counts_contourf_plotting(
                trial, projection_site_counts, trial_x, trial_y, cmap
            )
    # Movement behavior
    def vacancy_site_sd_time_recording(self):
        """
        Calculated sites where vacancy can visit during the simulation and final SD, also plotting the histogram of their value distribution
        """
        sites_visited_result = np.empty(self.trial_times)
        sd_result = np.empty(self.trial_times)
        time_result = np.empty(self.trial_times)
        for trial, (trial_x, trial_y, trial_z, sd) in enumerate(
            zip(self.x_columns, self.y_columns, self.z_columns, self.sd_columns)
        ):
            df = self.vacancy_record[[trial_x, trial_y, trial_z]].copy()
            df["count"] = 1
            site_couts = (
                df.groupby([trial_x, trial_y, trial_z])
                .agg({"count": len})
                .reset_index()
            )
            num_sites_visited = site_couts.shape[0]
            sites_visited_result[trial] = num_sites_visited
            sd_result[trial] = self.vacancy_record[sd].iloc[-1]
        sites_mean = np.mean(sites_visited_result)
        sites_median = np.median(sites_visited_result)
        sd_mean = np.mean(sd_result)
        sd_median = np.median(sd_result)
        with open("Site visited and MSD result.txt", mode="w") as fh:
            fh.write(
                f"{self.alloy_system}, {(sites_mean,sites_median,sd_mean,sd_median)}\n"
            )
            for site, sd in zip(sites_visited_result, sd_result):
                fh.write(f"{str(site)},{str(sd)}\n")
            fh.close()

        # 紀錄花費時間
        for trial, time_recording in enumerate(self.time_columns):
            time_spending = self.vacancy_record[time_recording].iloc[-1]
            time_result[trial] = time_spending
        time_mean = np.mean(time_result)
        time_median = np.median(time_result)
        with open("Vacancy time result.txt", mode="w") as fh:
            fh.write(f"{self.alloy_system}, {(time_mean,time_median)}\n")
            for time_recording in time_result:
                if time_recording != np.nan:
                    fh.write(f"{str(time_recording)}\n")
            fh.close()

        # 繪製site直方圖
        plot_site_detail = {
            "title": "Site Visit Distribution",
            "xlabel": "Counts",
            "ylabel": "Normalized Constant",
            "filename": "Site Visit Distribution.png",
        }
        self.result_histo_plotting(sites_visited_result, plot_site_detail)

        # 繪製sd直方圖
        plot_sd_detail = {
            "title": "SD Distribution",
            "xlabel": "SD",
            "ylabel": "Counts",
            "filename": "SD Distribution.png",
        }
        self.result_histo_plotting(sd_result, plot_sd_detail)

        # 繪製time直方圖
        plot_time_detail = {
            "title": "Time Distribution",
            "xlabel": "Time",
            "ylabel": "Counts",
            "filename": "Time Distribution.png",
        }
        self.result_histo_plotting(time_result, plot_time_detail)

        return sites_mean, sites_median, sd_mean, sd_median, time_mean, time_median

    def vacancy_backhop_probability(self):
        """
        Calculated vacancy back-hop probability where vacancy can visit during the simulation and final SD, also plotting the histogram of their value distribution
        """
        backhop_probability_record = np.empty(self.trial_times)
        for trial, (x, y, z) in enumerate(
            zip(self.x_columns, self.y_columns, self.z_columns)
        ):
            df = pd.DataFrame()
            difference = (
                self.vacancy_record[[x, y, z]]
                .diff()
                .fillna(value=self.vacancy_record.iloc[[0]])
            )
            shift = difference.shift(periods=1).fillna(0)
            df[x] = difference[x] + shift[x]
            df[y] = difference[y] + shift[y]
            df[z] = difference[z] + shift[z]
            backhop_probability_record[trial] = (
                df[(df[x] == 0) & (df[y] == 0) & (df[z] == 0)].shape[0]
                / self.vacancy_record.shape[0]
            )
        mean_probability = np.mean(backhop_probability_record)
        std_probability = np.std(backhop_probability_record)
        with open("Backhop Probability for Vacancy.txt", mode="w") as fh:
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
            "filename": "Back hop probability Distribution.png",
        }
        self.result_histo_plotting(backhop_probability_record, plot_backhoppro_detail)

        return backhop_probability_record, mean_probability, std_probability

    def vacancy_correlation_factor(self):
        sd = pd.DataFrame(self.vacancy_record[self.sd_columns])
        sd["MSD"] = sd.mean(axis=1, skipna=True)
        # Vacancy Correlation Factor
        if self.crystalstructure == "fcc":
            vacancy_f = sd.iloc[-1, -1] / (sd.shape[0] * 0.5)
        elif self.crystalstructure == "bcc":
            vacancy_f = sd.iloc[-1, -1] / (sd.shape[0] * 0.75)
        else:
            vacancy_f = sd.iloc[-1, -1] / (sd.shape[0] )    
        return vacancy_f 
    
    def result_histo_plotting(self, result, plot_time_detail):
        """
        Plotting results histogram.
        """
        bins = int(1 + 3.332 * np.log10(len(result)))
        plt.hist(result, bins=bins)
        plt.axvline(x=np.mean(result), color="r")
        plt.axvline(x=np.median(result), color="green")
        plt.title(plot_time_detail["title"])
        plt.xlabel(plot_time_detail["xlabel"])
        plt.ylabel(plot_time_detail["ylabel"])
        plt.tight_layout()
        plt.savefig(plot_time_detail["filename"])
        plt.close()

    # Diffusion behavior
    def vacancy_msd_timestep(self):
        """
        Plotting MSD as function of timestep
        """
        # Calculate mean square displacement(MSD) as function of timestep
        sd = pd.DataFrame(self.vacancy_record[self.sd_columns])
        sd["MSD"] = sd.mean(axis=1, skipna=True)
        # Vacancy Correlation Factor
        if self.crystalstructure == "fcc":
            vacancy_f = sd.iloc[-1, -1] / (sd.shape[0] * 0.5)
        elif self.crystalstructure == "bcc":
            vacancy_f = sd.iloc[-1, -1] / (sd.shape[0] * 0.75)
        else:
            vacancy_f = sd.iloc[-1, -1] / (sd.shape[0] )    
        # Plot MSD as funtion of timestep
        ax = sd.plot(y="MSD")
        os.chdir(os.path.join(self.maindir, "MSDPlotresult"))
        plt.title("MSD as function of timestep")
        plt.xlabel("Timestep")
        plt.ylabel("MSD")
        plt.savefig(f"MSD of {self.alloy_system} with correlation {vacancy_f}.png")
        plt.close()
        os.chdir(self.maindir)
        # Save file
        sd["MSD"].to_json("MSD_timestep.json")

        # linear fitting version
        linear_fit_timestep_result = pd.DataFrame()
        record = sd["MSD"]
        timestep = np.array(record.index).reshape(-1, 1)
        msd = np.array(record.fillna(0))
        avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
            timestep, msd
        )
        xfit = np.linspace(0, 100000, 20)
        yfit = avg_model.predict(xfit[:, np.newaxis])
        
        linear_fit_timestep_result["Timestep"] = xfit
        linear_fit_timestep_result["Fitted Result"] = yfit
        linear_fit_timestep_result.to_json("Linear Fit Timestep.json")
        os.chdir(self.maindir)
        return

    def vacancy_real_msd_timestep(self):
        sd = pd.DataFrame(self.vacancy_record[self.sd_columns])
        if self.crystalstructure == "fcc":
            sd["MSD"] = sd.mean(axis=1, skipna=True) * ((0.35 * 10e-9) ** 2)
        elif self.crystalstructure == "bcc":
            sd["MSD"] = sd.mean(axis=1, skipna=True) * ((0.32 * 10e-9) ** 2)
        else:
            sd["MSD"] = sd.mean(axis=1, skipna=True) * ((0.30 * 10e-9) ** 2)   
        # Plot MSD as funtion of timestep
        os.chdir(os.path.join(self.maindir, "MSDPlotresult"))
        plt.title("MSD as function of timestep")
        plt.xlabel("Timestep")
        plt.ylabel("MSD(nm2)")
        plt.savefig(f"Real_MSD of {self.alloy_system}.png")
        plt.close()
        os.chdir(self.maindir)
        sd["MSD"].to_json("MSD_real_timestep.json")

        # linear fitting version
        linear_fit_timestep_result = pd.DataFrame()
        record = sd["MSD"]
        timestep = np.array(record.index).reshape(-1, 1)
        msd = np.array(record.fillna(0))
        avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
            timestep, msd
        )
        xfit = np.linspace(0, 100000, 20)
        yfit = avg_model.predict(xfit[:, np.newaxis])
        
        linear_fit_timestep_result["Timestep"] = xfit
        linear_fit_timestep_result["Fitted Result"] = yfit
        linear_fit_timestep_result.to_json("Linear Fit Real Timestep.json")
        os.chdir(self.maindir)
        return

    def vacancy_msd_time(self):
        time_min_trial = self.vacancy_record[self.time_columns].iloc[-1].idxmin()
        time_ref = self.vacancy_record[time_min_trial].values
        time_count = len(time_ref)

        result = {}
        for i, (trial_time, trial_sd) in enumerate(
            zip(self.time_columns, self.sd_columns)
        ):
            value_list = np.zeros(time_count)
            index_list = self.vacancy_record[trial_time].searchsorted(time_ref)
            index_list -= 1
            for j in range(len(index_list)):
                if index_list[j] < 0:
                    value_list[j] = np.nan
                else:
                    value_list[j] = self.vacancy_record.at[index_list[j], trial_sd]
            result[f"SD{i}"] = value_list
        result = pd.DataFrame(result, index=time_ref.flatten())
        result["MSD"] = result.mean(skipna=True, axis=1)
        df = pd.DataFrame(
            {
                f"Time_{self.alloy_system}": time_ref,
                f"MSD_{self.alloy_system}": result["MSD"].values,
            }
        )
        df.to_json(f"MSD_time.json")
        if not os.path.exists(r"MSDPlotresult"):
            os.makedirs(r"MSDPlotresult")
        os.chdir(os.path.join(self.maindir, "MSDPlotresult"))
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.plot(result.index, result["MSD"])
        plt.title("MSD ")
        plt.xlabel("Time(s)")
        plt.ylabel("MSD")
        plt.savefig(f"MSD as function of time of {self.alloy_system}.png")
        plt.close()

        # linear fitting version
        linear_fit_timestep_result = pd.DataFrame()
        record = result["MSD"]
        timestep = np.array(record.index).reshape(-1, 1)
        msd = np.array(record.fillna(0))
        avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
            timestep, msd
        )
        xfit = np.linspace(0, result.index[-1], 20)
        yfit = avg_model.predict(xfit[:, np.newaxis])
        
        linear_fit_timestep_result["Timestep"] = xfit
        linear_fit_timestep_result["Fitted Result"] = yfit
        linear_fit_timestep_result.to_json("Linear Fit Time.json")
        os.chdir(self.maindir)
        return
    
    def vacancy_real_msd_time(self):
        time_min_trial = self.vacancy_record[self.time_columns].iloc[-1].idxmin()
        time_ref = self.vacancy_record[time_min_trial].values
        time_count = len(time_ref)

        result = {}
        for i, (trial_time, trial_sd) in enumerate(
            zip(self.time_columns, self.sd_columns)
        ):
            value_list = np.zeros(time_count)
            index_list = self.vacancy_record[trial_time].searchsorted(time_ref)
            index_list -= 1
            for j in range(len(index_list)):
                if index_list[j] < 0:
                    value_list[j] = np.nan
                else:
                    value_list[j] = self.vacancy_record.at[index_list[j], trial_sd]
            result[f"SD{i}"] = value_list
        result = pd.DataFrame(result, index=time_ref.flatten())
        result["MSD"] = result.mean(skipna=True, axis=1)
        df = pd.DataFrame(
            {
                f"Time_{self.alloy_system}": time_ref,
                f"MSD_{self.alloy_system}": result["MSD"].values
                * ((0.35 * 10e-9) ** 2),
            }
        )
        df.to_json(f"MSD_real_time.json")
        if not os.path.exists(r"MSDPlotresult"):
            os.makedirs(r"MSDPlotresult")
        os.chdir(os.path.join(self.maindir, "MSDPlotresult"))
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.plot(result.index, result["MSD"])
        plt.title("MSD ")
        plt.xlabel("Time(s)")
        plt.ylabel("MSD(nm2)")
        plt.savefig(f"Real MSD as function of time of {self.alloy_system}.png")
        plt.close()

        # linear fitting version
        linear_fit_timestep_result = pd.DataFrame()
        record = result["MSD"]
        timestep = np.array(record.index).reshape(-1, 1)
        msd = np.array(record.fillna(0))
        avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
            timestep, msd
        )
        xfit = np.linspace(0, result.index[-1], 20)
        yfit = avg_model.predict(xfit[:, np.newaxis])
        
        linear_fit_timestep_result["Timestep"] = xfit
        linear_fit_timestep_result["Fitted Result"] = yfit
        linear_fit_timestep_result.to_json("Linear Fit Real Time.json")
        os.chdir(self.maindir)

        return

    # Diffusvity
    def vacancy_sMSD(self):
        print(f"Now processing calculating Linear fiteed MSD timestep...")
        # From MSD to fit
        record = pd.read_json("MSD_timestep.json", orient="index")
        timestep = np.array(record.index).reshape(-1, 1)
        msd = np.array(record.fillna(0))
        avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
            timestep, msd
        )
        max_time = timestep.max()
        max_msd = msd.max()
        xfit = np.linspace(0, 100000, 1000)
        yfit = avg_model.predict(xfit[:, np.newaxis])
        fit_score = avg_model.score(timestep, msd)
        avg_vacancy_timestep_diffusivity = avg_model.coef_[0]
        plt.plot(xfit, yfit)
        plt.text(0.75 * max_time, 0.5 * max_msd, f"R={fit_score:.4f}", fontsize=10)
        plt.xlabel("Timestep")
        plt.ylabel("MSD")
        plt.savefig(f"MSD Timestep ")
        plt.close()
        with open(f"MSDTimestepResult.txt", "w") as fh:
            fh.write(f"{avg_vacancy_timestep_diffusivity}\n")
            fh.write(f"{fit_score}\n")
        os.chdir(self.maindir)
        return

    def vacancy_diffusivity(self):
        print(f"Now processing calculating diffusivity ...")
        # From MSD to fit
        msd_df = pd.read_json(f"MSD_time.json")
        if not os.path.exists(r"simulationresult"):
            os.mkdir(r"simulationresult")
        os.chdir("simulationresult")
        avg_time = np.array(msd_df[f"Time_{self.alloy_system}"]).reshape(-1, 1)
        msd = np.array(msd_df[f"MSD_{self.alloy_system}"].fillna(0))
        avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
            avg_time, msd
        )
        fit_score = avg_model.score(avg_time, msd)
        avg_vacancy_diffusivity = avg_model.coef_[0]

        max_time = avg_time.max()
        max_msd = msd.max()
        xfit = np.linspace(0, max_time, 1000)
        yfit = avg_model.predict(xfit[:, np.newaxis])
        plt.scatter(avg_time, msd)
        plt.plot(xfit, yfit)
        plt.xlabel("Time")
        plt.ylabel("MSD")
        plt.savefig(f"Diffusivity from MSD fitting")
        plt.close()
        with open(f"MSDTimeResult.txt", "w") as fh:
            fh.write(f"{avg_vacancy_diffusivity}\n")
            fh.write(f"{fit_score}\n")
        os.chdir(self.maindir)
        return

    def run_analysis(self):
        self.vacancy_backhop_probability()
        # self.vacancy_trace_counts_plotting()
        self.vacancy_site_sd_time_recording()
        self.run_vacancy_msd()

    def run_vacancy_msd(self):
        print(f"Now processing MSD vs timestep ...")
        self.vacancy_msd_timestep()
        print(f"Now processing real MSD vs timestep ...")
        self.vacancy_real_msd_timestep()
        print(f"Now processing MSD vs time ...")
        self.vacancy_msd_time()
        print(f"Now processing real MSD vs time ...")
        self.vacancy_real_msd_time()
        print(f"Now processing MSD timestep and correaltion factor")
        self.vacancy_MSD_diffusivity()
        self.vacancy_MSD_linearfit_correlationfactor()


class CrystalAnalyzer:
    def __init__(self, path, crystalstructure, temp,hypo_lattice_constant,outputdir) -> None:
        self.crystalstructurepath = path
        self.crystalstructure = crystalstructure
        self.temp = temp
        self.hypo_lattice_constant = hypo_lattice_constant
        self.maindir = os.path.abspath(os.path.curdir)
        self.alloysystems = [
            "1A",
            "1Abnb0",
            "1Abnb1",
            "1Abwb0",
            "1Abwb1",
            "2A-nb0",
            "2A-nb0.1",
            "2A-nb0.2",
            "2A-nb0.3",
            "3A-nb0",
            "3A-nb0.1",
            "3A-nb0.2",
            "3A-nb0.3",
            "4A-nb0",
            "4A-nb0.1",
            "4A-nb0.2",
            "4A-nb0.3",
            "5A-nb0",
            "5A-nb0.1",
            "5A-nb0.2",
            "5A-nb0.3",
            "2A-wb0",
            "2A-wb0.1",
            "2A-wb0.2",
            "2A-wb0.3",
            "3A-wb0",
            "3A-wb0.1",
            "3A-wb0.2",
            "3A-wb0.3",
            "4A-wb0",
            "4A-wb0.1",
            "4A-wb0.2",
            "4A-wb0.3",
            "5A-wb0",
            "5A-wb0.1",
            "5A-wb0.2",
            "5A-wb0.3",
        ]
        self.output_dir = outputdir
        self.outputpath = outputdir
        self.stylelist = {
            "1A": {"symbol": "o", "color": "black", "linestyle": (0, (5, 5))},
            "1Abnb0": {"symbol": "o", "color": "dimgrey", "linestyle": (0, (5, 5))},
            "1Abnb1": {"symbol": "o", "color": "darkgrey", "linestyle": (0, (5, 5))},
            "1Abwb0": {"symbol": "o", "color": "silver", "linestyle": (0, (5, 5))},
            "1Abwb1": {"symbol": "o", "color": "gainsboro", "linestyle": (0, (5, 5))},
            "2A-nb0": {
                "symbol": "s",
                "color": "mediumslateblue",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-nb0.1": {
                "symbol": "p",
                "color": "mediumslateblue",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-nb0.2": {
                "symbol": "P",
                "color": "mediumslateblue",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-nb0.3": {
                "symbol": "D",
                "color": "mediumslateblue",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-wb0": {
                "symbol": "s",
                "color": "purple",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-wb0.1": {
                "symbol": "p",
                "color": "purple",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-wb0.2": {
                "symbol": "P",
                "color": "purple",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-wb0.3": {
                "symbol": "D",
                "color": "purple",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "3A-nb0": {"symbol": "s", "color": "cyan", "linestyle": (5, (10, 3))},
            "3A-nb0.1": {"symbol": "p", "color": "cyan", "linestyle": (5, (10, 3))},
            "3A-nb0.2": {"symbol": "P", "color": "cyan", "linestyle": (5, (10, 3))},
            "3A-nb0.3": {"symbol": "D", "color": "cyan", "linestyle": (5, (10, 3))},
            "3A-wb0": {"symbol": "s", "color": "navy", "linestyle": (5, (10, 3))},
            "3A-wb0.1": {"symbol": "p", "color": "navy", "linestyle": (5, (10, 3))},
            "3A-wb0.2": {"symbol": "P", "color": "navy", "linestyle": (5, (10, 3))},
            "3A-wb0.3": {"symbol": "D", "color": "navy", "linestyle": (5, (10, 3))},
            "4A-nb0": {"symbol": "s", "color": "lime", "linestyle": "dashdot"},
            "4A-nb0.1": {"symbol": "p", "color": "lime", "linestyle": "dashdot"},
            "4A-nb0.2": {"symbol": "P", "color": "lime", "linestyle": "dashdot"},
            "4A-nb0.3": {"symbol": "D", "color": "lime", "linestyle": "dashdot"},
            "4A-wb0": {"symbol": "s", "color": "green", "linestyle": "dashdot"},
            "4A-wb0.1": {"symbol": "p", "color": "green", "linestyle": "dashdot"},
            "4A-wb0.2": {"symbol": "P", "color": "green", "linestyle": "dashdot"},
            "4A-wb0.3": {"symbol": "D", "color": "green", "linestyle": "dashdot"},
            "5A-nb0": {
                "symbol": "s",
                "color": "orange",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-nb0.1": {
                "symbol": "p",
                "color": "orange",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-nb0.2": {
                "symbol": "P",
                "color": "orange",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-nb0.3": {
                "symbol": "D",
                "color": "orange",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-wb0": {
                "symbol": "s",
                "color": "red",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-wb0.1": {
                "symbol": "p",
                "color": "red",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-wb0.2": {
                "symbol": "P",
                "color": "red",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-wb0.3": {
                "symbol": "D",
                "color": "red",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
        }
        self.symbol = ["o", "^", "p", "D"]
        self.colors = [
            "Red",
            "blue",
            "Black",
            "Orange",
            "Yellow",
            "Green",
            "lime",
            "cyan",
            "blue",
            "Black",
        ]
        self.spacing=3
    # Statistics
    def result_norm_histo_plotting(self, record, detail):
        """
        Plotting histogram
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        for i, sample_num in enumerate([20, 50, 80, len(record)]):
            row = i // 2
            col = i % 2

            ax = axes[row, col]
            result = record[0:sample_num]


            meanresult, medianresult, stdresult = (
                np.mean(result),
                np.median(result),
                np.std(result),
            )
            interval = [meanresult + stdresult * i for i in range(-3, 4)]
            ax.hist(result, bins=interval, density=True)


            mu, sigma = np.mean(result), np.std(result)
            x = np.linspace(meanresult - 5 * stdresult, meanresult + 5 * stdresult, 100)
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -((x - mu) ** 2) / (2 * sigma**2)
            )
            ax.plot(x, pdf, "r", label="Normal Distribution")

            ax.set_title(detail["title"] + f"{sample_num}")
            ax.set_xlabel(detail["xlabel"])
            ax.set_ylabel(detail["ylabel"])

        plt.suptitle(detail["title"])
        plt.savefig(detail["filename"])
        plt.tight_layout()
        plt.close()
        os.chdir(detail["path"])

    def alloysys_result_norm_plotting(self):
        """
        Plotting histogram
        """

        for alloy in self.alloysystems:
            os.chdir(alloy)
            with open(f"Backhop Probability for Vacancy.txt") as fh:
                fh.readline()
                backhopprobability = []
                for trail_backhopprobability in fh.readlines():
                    backhopprobability.append(float(trail_backhopprobability))
                meanbackhoppro = round(np.mean(backhopprobability), 4)
                stdbackhoppro = round(np.std(backhopprobability), 4)
            interval = [meanbackhoppro + stdbackhoppro * i for i in range(-3, 4)]
            plt.hist(backhopprobability, bins=interval, density=True)
            os.chdir(self.maindir)
        plt.title("Backhopprobability Distribution")
        plt.xlabel("Backhop Probability")
        plt.ylabel("Counts")
        plt.tight_layout()
        os.chdir(self.output_dir)
        plt.savefig(f"All alloysystem backhoppro disturbution.png")
        plt.close()
        os.chdir(self.maindir)

        for alloy in self.alloysystems:
            os.chdir(alloy)
            with open(f"Site visited and MSD result.txt") as fh:
                imformation = fh.readline().split(",")
                trialsite = []
                for data in fh.readlines():
                    trialsite.append(float(data.split(",")[0]))
                meansite = round(np.mean(trialsite), 3)
                stdsite = round(np.std(trialsite), 3)
            interval = [meansite + stdsite * i for i in range(-3, 4)]
            plt.hist(trialsite, bins=interval, density=True)
            os.chdir(self.maindir)
        plt.title("Site  Distribution")
        plt.xlabel("Site Visited")
        plt.ylabel("Counts")
        plt.tight_layout()
        os.chdir(self.output_dir)
        plt.savefig(f"All alloysystem site visited disturbution.png")
        plt.close()
        os.chdir(self.maindir)

        for alloy in self.alloysystems:
            os.chdir(alloy)
            with open(f"Site visited and MSD result.txt") as fh:
                imformation = fh.readline().split(",")
                trialsd = []
                for data in fh.readlines():
                    trialsd.append(float(data.split(",")[1]))
                meansd = round(np.mean(trialsd), 3)
                stdsd = round(np.std(trialsd), 3)
            interval = [meansd + stdsd * i for i in range(-3, 4)]
            plt.hist(trialsd, bins=interval, density=True)
            os.chdir(self.maindir)
        plt.title("SD  Distribution")
        plt.xlabel("SD")
        plt.ylabel("Counts")
        plt.tight_layout()
        os.chdir(self.output_dir)
        plt.savefig(f"All alloysystem sd disturbution.png")
        plt.close()
        os.chdir(self.maindir)

        for alloy in self.alloysystems:
            os.chdir(alloy)
            with open(f"Vacancy time result.txt") as fh:
                fh.readline()
                timelist = []
                for trail_time in fh.readlines():
                    if trail_time != None:
                        timelist.append(float(trail_time))
                meantime = np.mean(timelist)
                stdtime = np.std(timelist)
            interval = [meantime + stdtime * i for i in range(-3, 4)]
            plt.hist(timelist, bins=interval)
            os.chdir(self.maindir)
        plt.title("Time Distribution")
        plt.xlabel("time")
        plt.ylabel("Counts")
        plt.tight_layout()
        os.chdir(self.output_dir)
        plt.savefig(f"All alloysystem time disturbution.png")
        plt.close()
        os.chdir(self.maindir)

    def alloysys_result_qq_plotting(self):
        """
        Plotting qqplotting in different alloy system
        """
        fig, ax1 = plt.subplots()
        for alloy in self.alloysystems:
            os.chdir(alloy)
            with open(f"Backhop Probability for Vacancy.txt") as fh:
                fh.readline()
                backhopprobability = []
                for trail_backhopprobability in fh.readlines():
                    backhopprobability.append(float(trail_backhopprobability))
            sm.qqplot(np.array(backhopprobability), fit=True, line="45", ax=ax1)
            os.chdir(self.maindir)
        plt.title("Backhopprobability Distribution")
        plt.xlabel("Normal theoretical quantiles")
        plt.ylabel("Normal data quantiles")
        plt.tight_layout()
        os.chdir(self.output_dir)
        plt.savefig(f"All alloysystem backhoppro qqploting.png")
        plt.close()
        os.chdir(self.maindir)

        fig, ax2 = plt.subplots()
        for alloy in self.alloysystems:
            os.chdir(alloy)
            with open(f"Site visited and MSD result.txt") as fh:
                imformation = fh.readline().split(",")
                trialsite = []
                for data in fh.readlines():
                    trialsite.append(float(data.split(",")[0]))
            sm.qqplot(np.array(trialsite), fit=True, line="45", ax=ax2)
            os.chdir(self.maindir)
        plt.title("Site Distribution")
        plt.xlabel("Normal theoretical quantiles")
        plt.ylabel("Normal data quantiles")
        plt.tight_layout()
        os.chdir(self.output_dir)
        plt.savefig(f"All alloysystem site visited qqploting.png")
        plt.close()
        os.chdir(self.maindir)

        fig, ax3 = plt.subplots()
        for alloy in self.alloysystems:
            os.chdir(alloy)
            with open(f"Site visited and MSD result.txt") as fh:
                imformation = fh.readline().split(",")
                trialsd = []
                for data in fh.readlines():
                    trialsd.append(float(data.split(",")[1]))
            sm.qqplot(np.array(trialsd), fit=True, line="45", ax=ax3)
            os.chdir(self.maindir)
        plt.title("SD Distribution")
        plt.xlabel("Normal theoretical quantiles")
        plt.ylabel("Normal data quantiles")
        plt.tight_layout()
        os.chdir(self.output_dir)
        plt.savefig(f"All alloysystem sd qqplotting.png")
        plt.close()
        os.chdir(self.maindir)

        fig, ax4 = plt.subplots()
        for alloy in self.alloysystems:
            os.chdir(alloy)
            with open(f"Vacancy time result.txt") as fh:
                fh.readline()
                timelist = []
                for trail_time in fh.readlines():
                    if trail_time != None:
                        timelist.append(float(trail_time))
            sm.qqplot(np.array(timelist), fit=True, line="45", ax=ax4)
            os.chdir(self.maindir)
        plt.title("Time Distribution")
        plt.xlabel("Normal theoretical quantiles")
        plt.ylabel("Normal data quantiles")
        plt.tight_layout()
        os.chdir(self.output_dir)
        plt.savefig(f"All alloysystem time qqplotting.png")
        plt.close()
        os.chdir(self.maindir)

    def stat_norm_distribution_plotting(self):
        """
        Plotting distributions of statistical test
        """
        if not os.path.exists(r"statresult/"):
            os.makedirs(r"statresult/Site")
            os.makedirs(r"statresult/SD")
            os.makedirs(r"statresult/Time")
        for alloy in self.alloysystems:
            print(f"Now in {alloy}")
            alloypath = os.path.join(self.maindir, alloy)
            os.chdir(alloypath)
            # bachhop probability
            with open("Backhop Probability for Vacancy.txt", mode="r") as fh:
                prolist = []
                for pro in fh.readlines()[1:]:
                    prolist.append(float(pro))
            plot_backhoppro_detail = {
                "path": alloypath,
                "title": "Back hop probability Distribution",
                "xlabel": "Probability",
                "ylabel": "Normalized Constant",
                "normfilepath": os.path.join(
                    self.maindir, "statresult", "backhoppro", "histogram"
                ),
                "filename": f"{alloy} backhop probability Distribution.png",
                "qqtitle": "Back hop probability Q-Q plot",
                "qqxlabel": "Normal theoretical quantiles",
                "qqylabel": "Normal data quantiles",
                "qqfilepath": os.path.join(
                    self.maindir, "statresult", "backhoppro", "qqplot"
                ),
                "qqfilename": f"{alloy} backhop probability QQ plot.png",
            }
            self.result_norm_histo_plotting(prolist, plot_backhoppro_detail)
            self.qq_plotting(prolist, plot_backhoppro_detail)

            # site
            with open(f"Site visited and MSD result.txt", mode="r") as fh:
                imformation = fh.readline().split(",")
                trialsite = []
                trialsd = []
                for data in fh.readlines():
                    trialsite.append(float(data.split(",")[0]))
                    trialsd.append(float(data.split(",")[1]))
                plot_site_detail = {
                    "path": alloypath,
                    "title": "Site Distribution",
                    "xlabel": "Probability",
                    "ylabel": "Normalized Constant",
                    "normfilepath": os.path.join(
                        self.maindir, "statresult", "site", "histogram"
                    ),
                    "filename": f"{alloy} site Distribution.png",
                    "qqtitle": "Site Q-Q plot",
                    "qqxlabel": "Normal theoretical quantiles",
                    "qqylabel": "Normal data quantiles",
                    "qqfilepath": os.path.join(
                        self.maindir, "statresult", "site", "qqplot"
                    ),
                    "qqfilename": f"{alloy} backhop site QQ plot.png",
                }
                self.result_norm_histo_plotting(trialsite, plot_site_detail)
                self.qq_plotting(trialsite, plot_site_detail)
                # sd
                plot_sd_detail = {
                    "path": alloypath,
                    "title": "SD Distribution",
                    "xlabel": "Probability",
                    "ylabel": "Normalized Constant",
                    "normfilepath": os.path.join(
                        self.maindir, "statresult", "sd", "histogram"
                    ),
                    "filename": f"{alloy} sd Distribution.png",
                    "qqtitle": "SD Q-Q plot",
                    "qqxlabel": "Normal theoretical quantiles",
                    "qqylabel": "Normal data quantiles",
                    "qqfilepath": os.path.join(
                        self.maindir, "statresult", "sd", "qqplot"
                    ),
                    "qqfilename": f"{alloy} sd QQ plot.png",
                }
                self.result_norm_histo_plotting(trialsd, plot_sd_detail)
                self.qq_plotting(trialsd, plot_sd_detail)
            # Time
            with open("Vacancy time result.txt", mode="r") as fh:
                timelist = []
                for time in fh.readlines()[1:]:
                    if time != None:
                        timelist.append(float(time.strip("\n")))
            plot_time_detail = {
                "path": alloypath,
                "title": "Time Distribution",
                "xlabel": "Probability",
                "ylabel": "Normalized Constant",
                "normfilepath": os.path.join(
                    self.maindir, "statresult", "time", "histogram"
                ),
                "filename": f"{alloy} time Distribution.png",
                "qqtitle": "Time Q-Q plot",
                "qqxlabel": "Normal theoretical quantiles",
                "qqylabel": "Normal data quantiles",
                "qqfilepath": os.path.join(
                    self.maindir, "statresult", "time", "qqplot"
                ),
                "qqfilename": f"{alloy} time QQ plot.png",
            }
            self.result_norm_histo_plotting(timelist, plot_time_detail)
            self.qq_plotting(timelist, plot_time_detail)

            os.chdir(self.maindir)

    def qq_plotting(self, record, detail):
        """
        Plotting qqplot to check normalbility
        """
        if not os.path.exists(detail["qqfilepath"]):
            os.makedirs(detail["qqfilepath"])
        os.chdir(detail["qqfilepath"])
        fig = plt.figure(figsize=(10, 8))
        for i, sample_num in enumerate([20, 50, 80, len(record)]):
            ax = fig.add_subplot(2, 2, i + 1)
            result = np.array(record[0:sample_num])
            sm.graphics.qqplot(result, fit=True, line="45", ax=ax)
            # 調整座標軸名稱等資訊
        plt.suptitle(detail["title"])
        plt.savefig(detail["qqfilename"])
        plt.tight_layout()
        # plt.show()
        plt.close()
        os.chdir(detail["path"])

    def comparison_scatter_plotting(self, result, detail):
        os.chdir(detail["filepath"])
        for i, x in enumerate(self.alloysystems):
            plt.plot(
                self.spacing * i,
                result[i],
                marker=self.stylelist[x]["symbol"],
                color=self.stylelist[x]["color"],
            )
        plt.ylabel(detail["ylabel"])
        plt.title(detail["title"])
        xticks = [i * self.spacing for i in range(len(self.alloysystems))]
        plt.xticks(xticks)
        xticklabels = [xi for xi in self.alloysystems]
        plt.gca().set_xticks(xticks)
        plt.axvline(x=self.spacing * 4.5, color="black", linestyle="dashed")
        plt.axvline(x=self.spacing * 20.5, color="black", linestyle="dashed")
        plt.gca().set_xticklabels(xticklabels, rotation=270, ha="center")
        if detail["YFormatter"] == True:
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if "Ylog" in detail:
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(detail["scatterfilename"])
        plt.close()
        os.chdir(self.maindir)

    def comparison_errorbar_plotting(self, result, stdresult, detail):
        os.chdir(detail["filepath"])
        for i, x in enumerate(self.alloysystems):
            plt.errorbar(
                self.spacing * i,
                result[i],
                stdresult[i],
                elinewidth=2,
                capsize=2,
                fmt="o",
                marker=self.stylelist[x]["symbol"],
                color=self.stylelist[x]["color"],
            )
        plt.ylabel(detail["ylabel"])
        plt.title(detail["title"])
        xticks = [i * self.spacing for i in range(len(self.alloysystems))]
        plt.xticks(xticks)
        xticklabels = [xi for xi in self.alloysystems]
        plt.gca().set_xticks(xticks)
        plt.axvline(x=self.spacing * 4.5, color="black", linestyle="dashed")
        plt.axvline(x=self.spacing * 20.5, color="black", linestyle="dashed")
        plt.gca().set_xticklabels(xticklabels, rotation=270, ha="center")
        if detail["YFormatter"] == True:
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if "Ylog" in detail:
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(detail["errorbarfilename"])
        plt.close()
        os.chdir(self.maindir)

    def plotting_detail(self, detail):
        os.chdir(self.outputpath)
        plt.ylabel(detail["ylabel"])
        if detail["ysetlim"] == True:
            plt.ylim(detail["ylim"])
        if detail["yScilim"] == True:
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.title(detail["title"])
        xticks = [i * self.spacing for i in range(len(self.alloysystems))]
        plt.xticks(xticks)
        xticklabels = [xi for xi in self.alloysystems]
        plt.gca().set_xticks(xticks)
        plt.gca().set_xticklabels(xticklabels, rotation=270, ha="center")
        plt.axvline(x=self.spacing * 4.5, color="black", linestyle="dashed")
        plt.axvline(x=self.spacing * 20.5, color="black", linestyle="dashed")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(detail["filename"])
        plt.close()
        os.chdir(self.maindir)

    def read_backhop_probability_result(
        self,
    ):
        with open(f"Backhop Probability for Vacancy.txt") as fh:
            fh.readline()
            backhopprobability = []
            for trail_backhopprobability in fh.readlines():
                backhopprobability.append(float(trail_backhopprobability))
            meanbackhoppro = round(np.mean(backhopprobability), 4)
            medianbackhoppro = round(np.median(backhopprobability), 4)
            stdbackhoppro = round(np.std(backhopprobability), 4)
        return meanbackhoppro, medianbackhoppro, stdbackhoppro

    def read_sitevisit_sd_vacancy_result(self):
        with open(f"Site visited and MSD result.txt") as fh:
            imformation = fh.readline().split(",")
            trialsite = []
            trialsd = []
            for data in fh.readlines():
                trialsite.append(float(data.split(",")[0]))
                trialsd.append(float(data.split(",")[1]))
            trialmeansites, trialmeansd = np.mean(trialsite), np.mean(trialsd)
            trialmediansites, trialmediansd = np.median(trialsite), np.median(trialsd)
            trialstdsites, trialstdsd = np.std(trialsite), np.std(trialsd)
            meansite = round(trialmeansites, 1)
            mediansite = round(trialmediansites, 1)
            stdsite = round(trialstdsites, 2)
            meansd = round(trialmeansd, 2)
            mediansd = round(trialmediansd, 1)
            stdsd = round(trialstdsd, 2)
        return meansite, mediansite, stdsite, meansd, mediansd, stdsd

    def read_time_result(self):
        with open(f"Vacancy time result.txt") as fh:
            fh.readline()
            timelist = []
            for trail_time in fh.readlines():
                if trail_time != None:
                    timelist.append(float(trail_time))
            meantime = np.mean(timelist)
            mediantime = np.median(timelist)
            stdtime = np.std(timelist)
        return meantime, mediantime, stdtime
    
    def read_correlationfactor_result(self, crystalstructure):
        msd_df = pd.read_json("MSD_timestep.json", orient="index")
        if crystalstructure == "fcc":
            return msd_df.iloc[-1, -1] / (msd_df.shape[0] * 0.5)
        elif crystalstructure == "sc":
            return msd_df.iloc[-1, -1] / (msd_df.shape[0])
        else:
            return msd_df.iloc[-1, -1] / (msd_df.shape[0] * 0.75)

    def read_jumpingfrequency_result(self):
        with open(f"Vacancy time result.txt") as fh:
            fh.readline()
            timelist = []
            for trail_time in fh.readlines():
                if trail_time != None:
                    timelist.append(float(trail_time))
        return 100000 / np.mean(timelist)
    
    
    def read_diffusivity_result(self):
        with open(f"MSDVacancyDiffusivity.txt", mode="r") as fh:
            fittedslope = float(fh.readline().strip("\n"))
            diffusivity = (1 / 6) * fittedslope
        return diffusivity
    
    #  Movement bahavior
    def vacancy_correlation_factor_comparision(self):
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            correlationfactorlist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                correlationfactor = self.read_correlationfactor_result(
                    self.crystalstructure[k]
                )
                correlationfactorlist.append(correlationfactor)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        correlationfactorlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        correlationfactorlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_correlationfactor_scatter_detail = {
            "ylabel": "Correlation factor",
            "ysetlim": True,
            "ylim": (0, 1.01),
            "yScilim": True,
            "title": "Vacancy Correlation factor in different alloy system",
            "filename": f"Scatter Vacancy Correlation factor at {self.temp}",
        }
        self.plotting_detail(plot_correlationfactor_scatter_detail)
        return

    def vacancy_backhopprobability_plot(self):
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            meanprolist = []
            stdprolist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                meanpro, medianpro, stdpro = self.read_backhop_probability_result()
                meanprolist.append(meanpro)
                stdprolist.append(stdpro)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.errorbar(
                        self.spacing * i,
                        meanprolist[i],
                        yerr=stdprolist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.errorbar(
                        self.spacing * i,
                        meanprolist[i],
                        yerr=stdprolist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_pro_error_detail = {
            "ylabel": "Backhop probability",
            "ysetlim": True,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "Vacancy backhop probability in different alloy system",
            "filename": "Error bar Vacancy backhopprobability",
        }
        self.plotting_detail(plot_pro_error_detail)

    def vacancy_site_plot(self):
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            meansitelist = []
            stdsitelist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                (
                    meansite,
                    mediansite,
                    stdsite,
                    meansd,
                    mediansd,
                    stdsd,
                ) = self.read_sitevisit_sd_vacancy_result()
                meansitelist.append(meansite)
                stdsitelist.append(stdsite)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.errorbar(
                        self.spacing * i,
                        meansitelist[i],
                        yerr=stdsitelist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.errorbar(
                        self.spacing * i,
                        meansitelist[i],
                        yerr=stdsitelist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_site_error_detail = {
            "ylabel": "Site",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "Site visited in different alloy system",
            "filename": "Error bar Site",
        }
        self.plotting_detail(plot_site_error_detail)

        # result scatter plotting
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            mediansitelist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                (
                    meansite,
                    mediansite,
                    stdsite,
                    meansd,
                    mediansd,
                    stdsd,
                ) = self.read_sitevisit_sd_vacancy_result()
                mediansitelist.append(mediansite)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        mediansitelist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        mediansitelist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_site_scatter_detail = {
            "ylabel": "Site",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "Site visited in different alloy system",
            "filename": "Scatter Site",
        }
        self.plotting_detail(plot_site_scatter_detail)
        return

    def sd_result_plotting(self):
        # result scatter plotting
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            meansdlist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                (
                    meansite,
                    mediansite,
                    stdsite,
                    meansd,
                    mediansd,
                    stdsd,
                ) = self.read_sitevisit_sd_vacancy_result()
                meansdlist.append(meansd)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        meansdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        meansdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_meansd_scatter_detail = {
            "ylabel": "SD",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "Mean SD in different alloy system",
            "filename": "Scatter mean SD",
        }
        self.plotting_detail(plot_meansd_scatter_detail)

        # Median SD scatter plot
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            mediansdlist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                (
                    meansite,
                    mediansite,
                    stdsite,
                    meansd,
                    mediansd,
                    stdsd,
                ) = self.read_sitevisit_sd_vacancy_result()
                mediansdlist.append(mediansd)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        mediansdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        mediansdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_mediansd_scatter_detail = {
            "ylabel": "SD",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "Median SD in different alloy system",
            "filename": "Scatter median SD",
        }
        self.plotting_detail(plot_mediansd_scatter_detail)

        # real_sd_result_plotting
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            meansdlist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                (
                    meansite,
                    mediansite,
                    stdsite,
                    meansd,
                    mediansd,
                    stdsd,
                ) = self.read_sitevisit_sd_vacancy_result()
                meansdlist.append(meansd * (float(self.hypo_lattice_constant[k]) ** 2))
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        meansdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        meansdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_meansd_scatter_detail = {
            "ylabel": "SD",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "MSD in different alloy system",
            "filename": "Scatter REAL mean SD",
        }
        self.plotting_detail(plot_meansd_scatter_detail)

        # real Median SD scatter plot
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            mediansdlist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                (
                    meansite,
                    mediansite,
                    stdsite,
                    meansd,
                    mediansd,
                    stdsd,
                ) = self.read_sitevisit_sd_vacancy_result()
                mediansdlist.append(
                    mediansd * (float(self.hypo_lattice_constant[k]) ** 2)
                )
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        mediansdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        mediansdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_mediansd_scatter_detail = {
            "ylabel": "SD",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "Median SD in different alloy system",
            "filename": "Scatter REAL median SD",
        }
        self.plotting_detail(plot_mediansd_scatter_detail)
        
        # normailze sd
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            meansdlist = []
            stdsdlist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                (
                    meansite,
                    mediansite,
                    stdsite,
                    meansd,
                    mediansd,
                    stdsd,
                ) = self.read_sitevisit_sd_vacancy_result()
                if k == 0:
                    meansdlist.append(meansd)
                    stdsdlist.append(stdsd)
                elif k == 1:
                    meansdlist.append(meansd / (0.75))
                    stdsdlist.append(stdsd / (0.75))
                else:
                    meansdlist.append(meansd / (0.5))
                    stdsdlist.append(stdsd / (0.5))
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.errorbar(
                        self.spacing * i,
                        meansdlist[i],
                        yerr=stdsdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.errorbar(
                        self.spacing * i,
                        meansdlist[i],
                        yerr=stdsdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_meansd_error_detail = {
            "ylabel": "SD",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "Normalize MSD in different alloy system",
            "filename": "Error bar Normalize mean SD",
        }
        self.plotting_detail(plot_meansd_error_detail)

        # result scatter plotting
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            meansdlist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                (
                    meansite,
                    mediansite,
                    stdsite,
                    meansd,
                    mediansd,
                    stdsd,
                ) = self.read_sitevisit_sd_vacancy_result()
                if k == 0:
                    meansdlist.append(meansd)
                    stdsdlist.append(stdsd)
                elif k == 1:
                    meansdlist.append(meansd / (0.75))
                    stdsdlist.append(stdsd / (0.75))
                else:
                    meansdlist.append(meansd / (0.5))
                    stdsdlist.append(stdsd / (0.5))
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        meansdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        meansdlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_meansd_scatter_detail = {
            "ylabel": "SD",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "Normalize MSD in different alloy system",
            "filename": "Scatter Normalize mean SD",
        }
        self.plotting_detail(plot_meansd_scatter_detail)
        
        return

    def vacancy_time_plot(self):
        # result errorbar plotting
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            meantimelist = []
            stdtimelist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                meantime, mediantime, stdtime = self.read_time_result()
                meantimelist.append(meantime)
                stdtimelist.append(stdtime)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.errorbar(
                        self.spacing * i,
                        meantimelist[i],
                        yerr=stdtimelist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.errorbar(
                        self.spacing * i,
                        meantimelist[i],
                        yerr=stdtimelist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_time_error_detail = {
            "ylabel": "Time",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "Time spending in different alloy system",
            "filename": "Error bar Time",
        }
        plt.yscale("log")
        self.plotting_detail(plot_time_error_detail)

        # result scatter plotting
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            meantimelist = []
            stdtimelist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                meantime, mediantime, stdtime = self.read_time_result()
                meantimelist.append(meantime)
                stdtimelist.append(stdtime)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        meantimelist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        meantimelist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plt.yscale("log")
        plot_time_scatter_detail = {
            "ylabel": "Time",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "Time spending in different alloy system",
            "filename": "Scatter Time",
        }
        self.plotting_detail(plot_time_scatter_detail)

    def jumpingfrequency_result_plotting(self):
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            jumpingfrequencylist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(crystalstructure_path, alloysys)
                os.chdir(alloy_path)
                jumpingfrequency = self.read_jumpingfrequency_result()
                jumpingfrequencylist.append(jumpingfrequency)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        jumpingfrequencylist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        jumpingfrequencylist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_jumpingfrequency_scatter_detail = {
            "ylabel": "Jumping frequency",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": True,
            "title": "jumping frequency in different alloy system",
            "filename": "Scatter jumpingfrequency",
        }
        plt.yscale("log")
        self.plotting_detail(plot_jumpingfrequency_scatter_detail)
        return

    # Diffusion behavior
    def vacancy_msd_timestep_linear_fitted_plotting(self):
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloy in self.alloysystems:
                os.chdir(alloy)
                record = pd.read_json("MSD_timestep.json", orient="index")
                timestep = np.array(record.index).reshape(-1, 1)
                msd = np.array(record.fillna(0))
                avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
                    timestep, msd
                )
                xfit = np.linspace(0, 100000, 20)
                yfit = avg_model.predict(xfit[:, np.newaxis])
                plt.plot(
                    xfit,
                    yfit,
                    marker=self.stylelist[alloy]["symbol"],
                    color=self.stylelist[alloy]["color"],
                )
                os.chdir(self.maindir)

            plt.title("MSD as function of timestep in different alloy system")
            plt.xlabel("Timestep")
            plt.ylabel("MSD")
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.savefig("Linear Fitted MSD timestep in different Alloy system.jpg")
        return

    def vacancy_real_msd_timestep_linear_fitted_plotting(self):
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloy in self.alloysystems:
                os.chdir(alloy)
                record = pd.read_json("MSD_real_timestep.json", orient="index")
                timestep = np.array(record.index).reshape(-1, 1)
                msd = np.array(record.fillna(0))
                avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
                    timestep, msd
                )
                xfit = np.linspace(0, 100000, 1000)
                yfit = avg_model.predict(xfit[:, np.newaxis])
                plt.plot(
                    xfit,
                    yfit,
                    linestyle=self.stylelist[alloy]["linestyle"],
                    color=self.stylelist[alloy]["color"],
                )
                os.chdir(self.maindir)

            plt.title("MSD as function of timestep in different alloy system")
            plt.xlabel("Timestep")
            plt.ylabel("MSD")
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.savefig("Linear Fitted Real MSD timestep in different Alloy system.jpg")
            plt.close()
            os.chdir(self.maindir)
            return

    def vacancy_msd_time_fitted_plotting(self):
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))  
            msd_record = pd.DataFrame()

            for alloysys in self.alloysystems:
                os.chdir(alloysys)
                msd_df = pd.read_json("MSD_time.json").fillna(0)
                # 進行 linear fit 數據處理
                avg_time = np.array(msd_df[f"Time_{alloysys}"]).reshape(-1, 1)
                msd = np.array(msd_df[f"MSD_{alloysys}"].fillna(0))
                avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
                    avg_time, msd
                )
                max_time = avg_time.max()
                xfit = np.linspace(0, max_time, 1000)
                yfit = avg_model.predict(xfit[:, np.newaxis])
                msd_record[f"Time_{alloysys}"] = xfit
                msd_record[f"MSD_{alloysys}"] = yfit

                os.chdir(self.maindir)
            time_column = [col for col in msd_record.columns if "Time" in col]
            msd_column = [col for col in msd_record.columns if "MSD" in col]

            for time_record, msd in zip(time_column, msd_column):
                alloysys = msd.split("_")[1]
                ax1.plot(
                    msd_record[time_record],
                    msd_record[msd],
                    label=alloysys,
                    marker=self.stylelist[alloysys]["symbol"],
                    color=self.stylelist[alloysys]["color"],
                )
                ax2.plot(
                    msd_record[time_record],
                    msd_record[msd],
                    label=alloysys,
                    marker=self.stylelist[alloysys]["symbol"],
                    color=self.stylelist[alloysys]["color"],
                )

            os.chdir(os.path.join(self.maindir, "simulationresult"))
            ax1.set_title("MSD in different alloy systems")
            ax1.set_xlim(left=0)
            ax1.set_ylim(bottom=0)
            ax1.set_xlabel("Time")
            ax1.set_ylabel("MSD")
            ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

            time_min = min(msd_record[time_column].iloc[-1])
            msd_min = min(msd_record[msd_column].iloc[-1])
            ax2.set_title("MSD in different alloy systems")
            ax2.set_ylim(0, msd_min)
            ax2.set_xlim(0, time_min)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("MSD")
            ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.tight_layout()
            plt.savefig(f"Linear Fitted MSD time in different Alloy system.png")
            plt.close()
            os.chdir(self.maindir)

        return

    def vacancy_real_msd_time_fitted_plotting(self):
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))  # 增加了一个子图ax3
            msd_record = pd.DataFrame()

            for alloysys in self.alloysystems:
                os.chdir(alloysys)
                msd_df = pd.read_json("MSD_time.json").fillna(0)
                # 進行 linear fit 數據處理
                avg_time = np.array(msd_df[f"Time_{alloysys}"]).reshape(-1, 1)
                msd = np.array(msd_df[f"MSD_{alloysys}"].fillna(0)) * (self.hypo_lattice_constant[k] ** 2)
                avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
                    avg_time, msd
                )
                max_time = avg_time.max()
                xfit = np.linspace(0, max_time, 20)
                yfit = avg_model.predict(xfit[:, np.newaxis])
                msd_record[f"Time_{alloysys}"] = xfit
                msd_record[f"MSD_{alloysys}"] = yfit

                os.chdir(self.maindir)
            time_column = [col for col in msd_record.columns if "Time" in col]
            msd_column = [col for col in msd_record.columns if "MSD" in col]

            for time_record, msd in zip(time_column, msd_column):
                alloysys = msd.split("_")[1]
                ax1.plot(
                    msd_record[time_record],
                    msd_record[msd],
                    label=alloysys,
                    marker=self.stylelist[alloysys]["symbol"],
                    color=self.stylelist[alloysys]["color"],
                )
                ax2.plot(
                    msd_record[time_record],
                    msd_record[msd],
                    label=alloysys,
                    marker=self.stylelist[alloysys]["symbol"],
                    color=self.stylelist[alloysys]["color"],
                )

                os.chdir(os.path.join(self.maindir, "simulationresult"))
                ax1.set_title("MSD in different alloy systems")
                ax1.set_xlim(left=0)
                ax1.set_ylim(bottom=0)
                ax1.set_xlabel("Time")
                ax1.set_ylabel("MSD")
                ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

                time_min = min(msd_record[time_column].iloc[-1])
                msd_min = min(msd_record[msd_column].iloc[-1])
                ax2.set_title("MSD in different alloy systems")
                ax2.set_ylim(0, msd_min)
                ax2.set_xlim(0, time_min)
                ax2.set_xlabel("Time")
                ax2.set_ylabel("MSD")
                ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                plt.tight_layout()
                plt.savefig(f"Linear Fitted Real MSD time in different Alloy system.png")
                plt.close()

        return

    def diffusivity_result_plotting(self):
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            diffusivitylist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(
                    crystalstructure_path, alloysys, "simulationresult"
                )
                os.chdir(alloy_path)
                slope = self.read_diffusivity_result()
                diffusivitylist.append(
                    slope * (float(self.hypo_lattice_constant[k]) ** 2)
                )
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        diffusivitylist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        diffusivitylist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_pro_scatter_detail = {
            "ylabel": "Diffusivity",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": True,
            "title": "Vacancy diffusivity in different alloy system",
            "filename": f"Scatter Vacancy diffusvity at {self.temp}",
        }
        self.plotting_detail(plot_pro_scatter_detail)
        return

    def vacancy_movementbehavior_comparision(self):
        # 先判斷 jumping frequency
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)            
            result = {}
            for alloy in self.alloysystems:
                os.chdir(alloy)
                result[alloy] = {}
                result[alloy]["MSD"] = self.read_sitevisit_sd_vacancy_result()[3]
                result[alloy]["Frequency"] = float(100000 / self.read_time_result()[0])
                os.chdir(self.maindir)

            result_df = pd.DataFrame.from_dict(result, orient="index")

            frequency_ref = result_df.at["1A", "Frequency"]
            msd_ref = result_df.at["1A", "MSD"]

            for alloy in result_df.index:
                plt.scatter(
                    result_df.at[alloy, "Frequency"] / frequency_ref,
                    result_df.at[alloy, "MSD"] / msd_ref,
                    marker=self.stylelist[alloy]["symbol"],
                    color=self.stylelist[alloy]["color"],
                )
            os.chdir(os.path.join(self.maindir, "simulationresult"))
            plt.ylim(0, 1.21)
            plt.xscale("log")
            plt.xlabel("Jumping frequency")
            plt.ylabel("MSD")
            plt.savefig(f"Vacancy Jumping frequency vs MSD.png")
            plt.close()
            os.chdir(self.maindir)

    # temp
    def activationenergy_result_plotting(self):
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            activationenergylist = []
            crystalstructure_path = os.path.join(
                self.maindir,
                f"{self.crystalstructure[k]}Lattice",
                "hyposystem",
                "Tempresult",
                "TempArrheniusAlloysystem",
            )
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                with open(f"Diffusion coefficient of {alloysys}.txt", mode="r") as fh:
                    activation_energy = float(fh.readline().strip("\n").split(":")[1])
                    activationenergylist.append(activation_energy)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        activationenergylist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        activationenergylist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
            os.chdir(self.maindir)
        plot_pro_scatter_detail = {
            "ylabel": "Activation Energy (eV)",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": True,
            "title": "Vacancy Activation Energy in different alloy system",
            "filename": "Scatter Vacancy Activation Energy",
        }
        self.plotting_detail(plot_pro_scatter_detail)
        return
    
    def prefactor_result_plotting(self):
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            prefactorlist = []
            crystalstructure_path = os.path.join(
                self.maindir,
                f"{self.crystalstructure[k]}Lattice",
                "hyposystem",
                "Tempresult",
                "TempArrheniusAlloysystem",
            )
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                with open(f"Diffusion coefficient of {alloysys}.txt", mode="r") as fh:
                    fh.readline()
                    prefactor = float(fh.readline().strip("\n").split(":")[1])
                    prefactorlist.append(prefactor)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        prefactorlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        prefactorlist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
            os.chdir(self.maindir)
        plot_pro_scatter_detail = {
            "ylabel": "Prefactor ",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": True,
            "title": "Vacancy prefactor in different alloy system",
            "filename": "Scatter Vacancy prefactor",
        }
        self.plotting_detail(plot_pro_scatter_detail)
        return

    def effective_diffusion_barrier(self):
        plt.figure(figsize=(10, 6))
        for k, crystalstructure in enumerate(self.crystalstructurepath):
            legend_added = False
            prefactor_D = (
                (self.hypo_lattice_constant[k] ** 2) * (8 + 4 * k) * 1e13 / 6
            )  # 1/6
            barrierenergylist = []
            crystalstructure_path = os.path.join(self.maindir, crystalstructure)
            os.chdir(crystalstructure_path)
            for alloysys in self.alloysystems:
                alloy_path = os.path.join(
                    crystalstructure_path, alloysys, "simulationresult"
                )
                os.chdir(alloy_path)
                slope = self.read_diffusivity_result()
                diffusivity = slope * (float(self.hypo_lattice_constant[k]) ** 2)
                os.chdir(os.path.join(crystalstructure_path, alloysys))
                correlation_factor = self.read_correlationfactor_result(
                    self.crystalstructure[k]
                )
                barrier_energy = (
                    -8.617
                    * 1e-5
                    * self.temp
                    * np.log(diffusivity / (correlation_factor * prefactor_D))
                )
                barrierenergylist.append(barrier_energy)
                os.chdir(crystalstructure_path)
            for i, x in enumerate(self.alloysystems):
                if not legend_added:
                    plt.scatter(
                        self.spacing * i,
                        barrierenergylist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                        label=self.crystalstructure[k],
                    )
                    legend_added = True
                else:
                    plt.scatter(
                        self.spacing * i,
                        barrierenergylist[i],
                        marker=self.symbol[k],
                        color=self.colors[k],
                    )
        plot_barrierenergy_scatter_detail = {
            "ylabel": "Effective diffusion barrier",
            "ysetlim": False,
            "ylim": (0, 0.9),
            "yScilim": False,
            "title": "Vacancy Effective diffusion barrierin different alloy system",
            "filename": f"Scatter Effective diffusion barrier at {self.temp}",
        }
        self.plotting_detail(plot_barrierenergy_scatter_detail)
        return



class TempAlloysystemAnalyzer:
    def __init__(self, crystalstructure, temppath, temp, outputpath) -> None:
        self.temppath = temppath
        self.crystalstructure = crystalstructure
        self.temp = temp
        self.maindir = os.path.abspath(os.path.curdir)
        self.outputpath = outputpath
        self.spacing = 0.2
        self.symbol = ["o", "^", "p", "D"]
        self.colors = [
            "Red",
            "Orange",
            "Yellow",
            "Green",
            "lime",
            "cyan",
            "blue",
            "Black",
        ]
        self.stylelist = {
            "1A": {"symbol": "o", "color": "black", "linestyle": (0, (5, 5))},
            "1Abnb0": {"symbol": "o", "color": "dimgrey", "linestyle": (0, (5, 5))},
            "1Abnb1": {"symbol": "o", "color": "darkgrey", "linestyle": (0, (5, 5))},
            "1Abwb0": {"symbol": "o", "color": "silver", "linestyle": (0, (5, 5))},
            "1Abwb1": {"symbol": "o", "color": "gainsboro", "linestyle": (0, (5, 5))},
            "2A-nb0": {
                "symbol": "s",
                "color": "mediumslateblue",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-nb0.1": {
                "symbol": "p",
                "color": "mediumslateblue",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-nb0.2": {
                "symbol": "P",
                "color": "mediumslateblue",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-nb0.3": {
                "symbol": "D",
                "color": "mediumslateblue",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-wb0": {
                "symbol": "s",
                "color": "purple",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-wb0.1": {
                "symbol": "p",
                "color": "purple",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-wb0.2": {
                "symbol": "P",
                "color": "purple",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "2A-wb0.3": {
                "symbol": "D",
                "color": "purple",
                "linestyle": (0, (3, 5, 1, 5)),
            },
            "3A-nb0": {"symbol": "s", "color": "cyan", "linestyle": (5, (10, 3))},
            "3A-nb0.1": {"symbol": "p", "color": "cyan", "linestyle": (5, (10, 3))},
            "3A-nb0.2": {"symbol": "P", "color": "cyan", "linestyle": (5, (10, 3))},
            "3A-nb0.3": {"symbol": "D", "color": "cyan", "linestyle": (5, (10, 3))},
            "3A-wb0": {"symbol": "s", "color": "navy", "linestyle": (5, (10, 3))},
            "3A-wb0.1": {"symbol": "p", "color": "navy", "linestyle": (5, (10, 3))},
            "3A-wb0.2": {"symbol": "P", "color": "navy", "linestyle": (5, (10, 3))},
            "3A-wb0.3": {"symbol": "D", "color": "navy", "linestyle": (5, (10, 3))},
            "4A-nb0": {"symbol": "s", "color": "lime", "linestyle": "dashdot"},
            "4A-nb0.1": {"symbol": "p", "color": "lime", "linestyle": "dashdot"},
            "4A-nb0.2": {"symbol": "P", "color": "lime", "linestyle": "dashdot"},
            "4A-nb0.3": {"symbol": "D", "color": "lime", "linestyle": "dashdot"},
            "4A-wb0": {"symbol": "s", "color": "green", "linestyle": "dashdot"},
            "4A-wb0.1": {"symbol": "p", "color": "green", "linestyle": "dashdot"},
            "4A-wb0.2": {"symbol": "P", "color": "green", "linestyle": "dashdot"},
            "4A-wb0.3": {"symbol": "D", "color": "green", "linestyle": "dashdot"},
            "5A-nb0": {
                "symbol": "s",
                "color": "tomato",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-nb0.1": {
                "symbol": "p",
                "color": "tomato",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-nb0.2": {
                "symbol": "P",
                "color": "tomato",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-nb0.3": {
                "symbol": "D",
                "color": "tomato",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-wb0": {
                "symbol": "s",
                "color": "red",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-wb0.1": {
                "symbol": "p",
                "color": "red",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-wb0.2": {
                "symbol": "P",
                "color": "red",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
            "5A-wb0.3": {
                "symbol": "D",
                "color": "red",
                "linestyle": (0, (3, 5, 1, 5, 1, 5)),
            },
        }
        
        self.alloysystems = [
            "1A",
            "1Abnb0",
            "1Abnb1",
            "1Abwb0",
            "1Abwb1",
            "2A-nb0",
            "2A-nb0.1",
            "2A-nb0.2",
            "2A-nb0.3",
            "3A-nb0",
            "3A-nb0.1",
            "3A-nb0.2",
            "3A-nb0.3",
            "4A-nb0",
            "4A-nb0.1",
            "4A-nb0.2",
            "4A-nb0.3",
            "5A-nb0",
            "5A-nb0.1",
            "5A-nb0.2",
            "5A-nb0.3",
            "2A-wb0",
            "2A-wb0.1",
            "2A-wb0.2",
            "2A-wb0.3",
            "3A-wb0",
            "3A-wb0.1",
            "3A-wb0.2",
            "3A-wb0.3",
            "4A-wb0",
            "4A-wb0.1",
            "4A-wb0.2",
            "4A-wb0.3",
            "5A-wb0",
            "5A-wb0.1",
            "5A-wb0.2",
            "5A-wb0.3",
        ]

    def comparison_scatter_plotting(self, result, detail):
        if not os.path.exists(detail["filepath"]):
            os.makedirs(detail["filepath"])
        os.chdir(detail["filepath"])
        for i, x in enumerate(self.alloysystems):
            plt.scatter(
                self.spacing * i,
                result[i],
                marker=self.stylelist[x]["symbol"],
                color=self.stylelist[x]["color"],
            )
        plt.ylabel(detail["ylabel"])
        plt.title(detail["title"])
        xticks = [i * self.spacing for i in range(len(self.alloysystems))]
        plt.xticks(xticks)
        xticklabels = [xi for xi in self.alloysystems]
        plt.gca().set_xticks(xticks)
        plt.axvline(x=self.spacing * 4.5, color="black", linestyle="dashed")
        plt.axvline(x=self.spacing * 20.5, color="black", linestyle="dashed")
        plt.gca().set_xticklabels(xticklabels, rotation=270, ha="center")
        if detail["YFormatter"] == True:
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.tight_layout()
        plt.savefig(detail["scatterfilename"])
        plt.close()
        os.chdir(self.maindir)

    def comparision_temp_results_plotting(self, result, detail):
        if not os.path.exists(detail["filepath"]):
            os.makedirs(detail["filepath"])
        ref = result.at["1A", 1000]
        fig, ax = plt.subplots(figsize=(9, 4))
        if detail["Normalized"]:
            im = ax.imshow(result.transpose() / ref, cmap="rainbow")
        else:
            im = ax.imshow(result.transpose(), cmap="rainbow")
        ax.set_xticks(np.arange(len(self.alloysystems)), labels=self.alloysystems)
        ax.set_yticks(np.arange(len(self.temp)), labels=self.temp)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        # 添加數值到上面
        """
        for i, temperature in enumerate(self.temp):
            for j, alloysystem in enumerate(self.alloysystems):
                text = ax.text(
                    j,
                    i,
                    f"{site_result_df.at[alloysystem, temperature] / ref :.2f}",
                    ha="center",
                    va="center",
                    color="w",
                )
        """
        cbar = plt.colorbar(im, fraction=0.015, pad=0.02)
        os.chdir(detail["filepath"])
        plt.ylabel(detail["ylabel"])
        plt.title(detail["title"])
        plt.tight_layout()
        # plt.show()
        plt.savefig(detail["filename"])
        plt.close()
        os.chdir(self.maindir)
        return

    def read_sitevisit_sd_vacancy_result(self):
        with open(f"Site visited and MSD result.txt") as fh:
            imformation = fh.readline().split(",")
            trialsite = []
            trialsd = []
            for data in fh.readlines():
                trialsite.append(float(data.split(",")[0]))
                trialsd.append(float(data.split(",")[1]))
            trialmeansites, trialmeansd = np.mean(trialsite), np.mean(trialsd)
            trialmediansites, trialmediansd = np.median(trialsite), np.median(trialsd)
            trialstdsites, trialstdsd = np.std(trialsite), np.std(trialsd)
            meansite = round(trialmeansites, 1)
            mediansite = round(trialmediansites, 1)
            stdsite = round(trialstdsites, 2)
            meansd = round(trialmeansd, 2)
            mediansd = round(trialmediansd, 1)
            stdsd = round(trialstdsd, 2)
        return meansite, mediansite, stdsite, meansd, mediansd, stdsd

    def read_backhop_probability_result(self):
        with open(f"Backhop Probability for Vacancy.txt") as fh:
            fh.readline()
            backhopprobability = []
            for trail_backhopprobability in fh.readlines():
                backhopprobability.append(float(trail_backhopprobability))
            meanbackhoppro = round(np.mean(backhopprobability), 4)
        return meanbackhoppro

    def read_time_result(self):
        with open(f"Vacancy time result.txt") as fh:
            fh.readline()
            timelist = []
            for trail_time in fh.readlines():
                if trail_time != None:
                    timelist.append(float(trail_time))
        return np.mean(timelist)

    def read_diffusivity_result(self):
        hypo_lattice_constant = 0.35 * 1e-9
        with open(f"MSDVacancyDiffusivity.txt", mode="r") as fh:
            fittedslope = float(fh.readline().strip("\n"))
            diffusivity = (1 / 6) * fittedslope * (hypo_lattice_constant**2)

        return diffusivity

    def read_correlationfactor_result(self):
        msd_df = pd.read_json("MSD_timestep.json", orient="index")
        return msd_df.iloc[-1, -1] / (msd_df.shape[0] * 0.5)

    def read_jumpingfreqency_result(self):
        with open(f"Vacancy time result.txt") as fh:
            fh.readline()
            timelist = []
            for trail_time in fh.readlines():
                if trail_time != None:
                    timelist.append(float(trail_time))
        return 100000 / np.mean(timelist)

    def arrhenius_diffusivity(self):
        if self.crystalstructure == "fcc":
            hypo_lattice_constant = 0.35 * 1e-9  
        elif self.crystalstructure == "bcc":
            hypo_lattice_constant = 0.32 * 1e-9
        else:
            hypo_lattice_constant = 0.30 * 1e-9

        for alloysys in self.alloysystems:
            templist = []
            tempdiffusivitylist = []
            for temp, temppath in zip(self.temp, self.temppath):
                os.chdir(
                    os.path.join(self.maindir, temppath, alloysys, "simulationresult")
                )
                with open(f"MSDVacancyDiffusivity.txt", mode="r") as fh:
                    fittedslope = float(fh.readline().strip("\n"))
                inverse_temperature = 1 / float(temp)
                diffusivity = (1 / 6) * fittedslope * (hypo_lattice_constant**2)

                templist.append(float(inverse_temperature))
                tempdiffusivitylist.append(diffusivity)
                os.chdir(self.maindir)

            if not os.path.exists(
                os.path.join(self.outputpath, "TempArrheniusAlloysystem")
            ):
                os.makedirs(os.path.join(self.outputpath, "TempArrheniusAlloysystem"))
            os.chdir(os.path.join(self.outputpath, "TempArrheniusAlloysystem"))
            # 繪製阿瑞尼斯關係圖
            plt.scatter(templist, tempdiffusivitylist, label="Actual")
            # Fit 數據找出活化能與D0
            temp_x = np.array(templist).reshape(-1, 1)
            ln_tempd_y = np.log(tempdiffusivitylist)
            model = LinearRegression().fit(temp_x, ln_tempd_y)
            x_fit = np.linspace(templist[0], templist[-1], 10)
            y_fit = model.predict(x_fit[:, np.newaxis])
            fit_score = model.score(temp_x, ln_tempd_y)
            activation_energy = -model.coef_[0] * 8.617 * 1e-5
            d0 = np.exp(model.intercept_)
            plt.plot(x_fit, np.exp(y_fit), linestyle="-.", label="Fitted")
            plt.xlabel("1/T (1/K)")
            plt.ylabel("Diffusivity")
            plt.yscale("log")
            # plt.ylim([1e-13, 1e-8])
            plt.legend()
            plt.savefig(f"Arrhenius Relationship of {alloysys}.jpg")
            plt.close()
            with open(f"Diffusion coefficient of {alloysys}.txt", mode="w") as fh:
                fh.write(f"Activation Energy : {activation_energy}\n")
                fh.write(f"D0 : {d0}\n")
            os.chdir(self.maindir)
        return

    def activation_energy_comparision(self):
        os.chdir(os.path.join(self.outputpath, "TempArrheniusAlloysystem"))
        activation_energy_list = []
        for alloy in self.alloysystems:
            with open(f"Diffusion coefficient of {alloy}.txt", mode="r") as fh:
                activation_energy = float(fh.readline().strip("\n").split(":")[1])
                activation_energy_list.append(activation_energy)

        plot_details = {
            "title": "Vacancy Activation Energy in different alloy system",
            "ylabel": "Activation energy(eV)",
            "scatterfilename": r"scatter_Vacancy Activation Energy.png",
            "YFormatter": False,
            "filepath": os.path.join(self.outputpath, "TempResult"),
            "errorbarfilename": r"errorbar_Vacancy Activation Energy.png",
        }
        self.comparison_scatter_plotting(
            activation_energy_list,
            plot_details,
        )
        return

    def prefactor_comparision(self):
        os.chdir(os.path.join(self.outputpath, "TempArrheniusAlloysystem"))
        prefactor_list = []
        for alloy in self.alloysystems:
            with open(f"Diffusion coefficient of {alloy}.txt", mode="r") as fh:
                fh.readline()
                prefactor = float(fh.readline().strip("\n").split(":")[1])
                prefactor_list.append(prefactor)

        plot_details = {
            "title": "Vacancy Prefactor in different alloy system",
            "ylabel": "D0",
            "scatterfilename": r"scatter_Vacancy Prefactor.png",
            "YFormatter": True,
            "filepath": os.path.join(self.outputpath, "TempResult"),
            "errorbarfilename": r"errorbar_Vacancy Prefactor.png",
        }
        self.comparison_scatter_plotting(
            prefactor_list,
            plot_details,
        )
        return

    def vacancy_effctive_diffusion_barrier(self):

        if self.crystalstructure == "fcc":
            hypo_lattice_constant = 0.35 * 1e-9  
            prefactor_D = (hypo_lattice_constant**2) * 12 * 1e13 / 6
        elif self.crystalstructure == "bcc":
            hypo_lattice_constant = 0.32 * 1e-9
            prefactor_D = (hypo_lattice_constant**2) * 8 * 1e13 / 6
        else:
            hypo_lattice_constant = 0.30 * 1e-9
            prefactor_D = (hypo_lattice_constant**2) * 6 * 1e13 / 6
        
        barrier_energy_result = {}
        for temperature, temperaturpath in zip(self.temp, self.temppath):
            barrier_energy_result[temperature] = {}
            for alloysys in self.alloysystems:
                # 讀取該溫度下該系統擴散係數
                os.chdir(
                    os.path.join(
                        self.maindir, temperaturpath, alloysys, "simulationresult"
                    )
                )
                diffusivity = self.read_diffusivity_result()
                os.chdir(self.maindir)
                # 讀取該溫度下該系統correlaiton factor
                os.chdir(os.path.join(self.maindir, temperaturpath, alloysys))
                correlation_factor = self.read_correlationfactor_result()
                barrier_energy = (
                    -8.617
                    * 1e-5
                    * temperature
                    * np.log(diffusivity / (correlation_factor * prefactor_D))
                )
                barrier_energy_result[temperature][alloysys] = barrier_energy
        barrier_energy_result_df = pd.DataFrame.from_dict(
            barrier_energy_result, orient="columns"
        )
        # 分別繪製不同溫度下各系統分布
        for temp in barrier_energy_result_df.columns:
            plot_details = {
                "title": f"Vacancy Diffusion Barrier in different alloy system at {temp}K",
                "ylabel": "Barrier(eV)",
                "scatterfilename": f"scatter_Vacancy Diffusion Barrier at {temp}.png",
                "YFormatter": False,
                "filepath": os.path.join(
                    self.outputpath, "effective barrier", f"{temp}K"
                ),
                "errorbarfilename": f"errorbar_Vacancy Diffusion Barrier at {temp}.png",
            }
            self.comparison_scatter_plotting(
                barrier_energy_result_df[temp],
                plot_details,
            )
        # 分別繪製不同系統下各溫度分布
        for i, alloysys in enumerate(barrier_energy_result_df.index):
            alloysys_result = barrier_energy_result_df.iloc[i, :]
            for k, temp in enumerate(self.temp):
                plt.scatter(
                    self.spacing * k,
                    alloysys_result[temp],
                    marker=self.stylelist[alloysys]["symbol"],
                    color=self.stylelist[alloysys]["color"],
                )
        if not os.path.exists(
            os.path.join(self.outputpath, "effective barrier", "alloysystem")
        ):
            os.makedirs(
                os.path.join(self.outputpath, "effective barrier", "alloysystem")
            )
        os.chdir(os.path.join(self.outputpath, "effective barrier", "alloysystem"))
        plt.ylabel("Barrier(eV)")
        plt.title(f"Vacancy Diffusion Barrier in all alloysystem")
        xticks = [i * self.spacing for i in range(len(self.alloysystems))]
        plt.xticks(xticks)
        xticklabels = [xi for xi in self.alloysystems]
        plt.gca().set_xticks(xticks)
        plt.axvline(x=self.spacing * 4.5, color="black", linestyle="dashed")
        plt.axvline(x=self.spacing * 20.5, color="black", linestyle="dashed")
        plt.gca().set_xticklabels(xticklabels, rotation=270, ha="center")
        plt.tight_layout()
        plt.savefig(f"All alloy temp Vacancy Diffusion Barrier in {alloysys}.png")
        plt.close()
        os.chdir(self.maindir)
        return

    def temperature_diffusivity_comparison_plotting(self):
        if self.crystalstructure == "fcc":
            hypo_lattice_constant = 0.35 * 1e-9  
        elif self.crystalstructure == "bcc":
            hypo_lattice_constant = 0.32 * 1e-9
        else:
            hypo_lattice_constant = 0.30 * 1e-9
        plt.subplot()
        for alloy in self.alloysystems:
            diffusivitylist = []
            temperaturelist = []
            for temperature, temppath in zip(self.temp, self.temppath):
                os.chdir(
                    os.path.join(self.maindir, temppath, alloy, "simulationresult")
                )
                with open(f"MSDVacancyDiffusivity.txt", mode="r") as fh:
                    inverse_temperature = 1000 / float(temperature)
                    fittedslope = float(fh.readline().strip("\n"))
                    diffusivity = (1 / 6) * fittedslope * (hypo_lattice_constant**2)
                diffusivitylist.append(diffusivity)
                temperaturelist.append(inverse_temperature)
            plt.plot(
                temperaturelist,
                diffusivitylist,
                marker=self.stylelist[alloy]["symbol"],
                color=self.stylelist[alloy]["color"],
            )
            os.chdir(self.maindir)
        if not os.path.exists(os.path.join(self.outputpath, "TempResult")):
            os.makedirs(os.path.join(self.outputpath, "TempResult"))
        os.chdir(os.path.join(self.outputpath, "TempResult"))
        plt.xlabel("1/T (1/K)")
        plt.ylabel("Diffusivity")
        plt.yscale("log")
        # plt.ylim([1e-13, 1e-8])
        # plt.legend()
        plt.savefig(f"Arrhenius Relationship of allsystem.jpg")
        plt.close()
        # plt.show()

        return

    def vacancy_site_temp_comparison(self):
        site_result = {}
        for temperature, temperaturepath in zip(self.temp, self.temppath):
            site_result[temperature] = {}
            for alloy in self.alloysystems:
                os.chdir(os.path.join(self.maindir, temperaturepath, alloy))
                alloy_result = self.read_sitevisit_sd_vacancy_result()
                site_result[temperature][alloy] = alloy_result[0]
                os.chdir(os.path.join(self.maindir, temperaturepath))
            os.chdir(self.maindir)
        site_result_df = pd.DataFrame.from_dict(site_result, orient="columns")

        plot_details = {
            "title": "Site Visited in different alloy system at different temperature",
            "ylabel": "Temperature",
            "filename": r"Site Visited.png",
            "Normalized": True,
            "filepath": os.path.join(self.outputpath, "tempsimulationresult"),
        }

        self.comparision_temp_results_plotting(site_result_df, plot_details)
        return

    def vacancy_backhop_probability_temp_comparison(self):
        backhop_probability_result = {}
        for temperature, temperaturepath in zip(self.temp, self.temppath):
            os.chdir(temperaturepath)
            backhop_probability_result[temperature] = {}
            for alloy in self.alloysystems:
                os.chdir(os.path.join(self.maindir, temperaturepath, alloy))
                alloy_result = self.read_backhop_probability_result()
                backhop_probability_result[temperature][alloy] = alloy_result
                os.chdir(os.path.join(self.maindir, temperaturepath))
            os.chdir(self.maindir)
        backhop_probability_result_df = pd.DataFrame.from_dict(
            backhop_probability_result, orient="columns"
        )

        plot_details = {
            "title": "Vacancy Backhopprobaility in different alloy system at different temperature",
            "ylabel": "Temperature",
            "filename": r"Vacancy Backhopprobaility.png",
            "Normalized": False,
            "filepath": os.path.join(self.outputpath, "tempsimulationresult"),
        }

        self.comparision_temp_results_plotting(
            backhop_probability_result_df, plot_details
        )

        return

    def vacancy_time_temp_comparison(self):
        time_result = {}
        for temperature, temperaturepath in zip(self.temp, self.temppath):
            os.chdir(temperaturepath)
            time_result[temperature] = {}
            for alloy in self.alloysystems:
                os.chdir(os.path.join(self.maindir, temperaturepath, alloy))
                alloy_result = np.log(self.read_time_result())
                time_result[temperature][alloy] = alloy_result
                os.chdir(os.path.join(self.maindir, temperaturepath))
            os.chdir(self.maindir)
        time_result_df = pd.DataFrame.from_dict(time_result, orient="columns")

        plot_details = {
            "title": "Vacancy time spending in different alloy system at different temperature",
            "ylabel": "Temperature",
            "filename": r"Vacancy time spending.png",
            "Normalized": False,
            "filepath": os.path.join(self.outputpath, "tempsimulationresult"),
        }

        self.comparision_temp_results_plotting(time_result_df, plot_details)

    def vacancy_diffusivity_temp_comparison(self):
        diffusivity_result = {}
        for temperature, temperaturepath in zip(self.temp, self.temppath):
            os.chdir(temperaturepath)
            diffusivity_result[temperature] = {}
            for alloy in self.alloysystems:
                os.chdir(
                    os.path.join(
                        self.maindir, temperaturepath, alloy, "simulationresult"
                    )
                )
                alloy_result = np.log(self.read_diffusivity_result())
                diffusivity_result[temperature][alloy] = alloy_result
                os.chdir(os.path.join(self.maindir, temperaturepath))
            os.chdir(self.maindir)
        diffusivity_result_df = pd.DataFrame.from_dict(
            diffusivity_result, orient="columns"
        )

        plot_details = {
            "title": "Vacancy Diffusivity in different alloy system at different temperature",
            "ylabel": "Temperature",
            "filename": r"Vacancy Diffusivity.png",
            "Normalized": False,
            "filepath": os.path.join(self.outputpath, "tempsimulationresult"),
        }

        self.comparision_temp_results_plotting(diffusivity_result_df, plot_details)

        return

    def vacancy_correlationfactor_temp_comparison(self):
        correlationfactor_result = {}
        for temperature, temperaturepath in zip(self.temp, self.temppath):
            correlationfactor_result[temperature] = {}
            for alloy in self.alloysystems:
                os.chdir(os.path.join(self.maindir, temperaturepath, alloy))
                alloy_result = self.read_correlationfactor_result()
                correlationfactor_result[temperature][alloy] = alloy_result
                os.chdir(os.path.join(self.maindir, temperaturepath))
            os.chdir(self.maindir)
        correlationfactor_result_df = pd.DataFrame.from_dict(
            correlationfactor_result, orient="columns"
        )

        plot_details = {
            "title": "Vacancy Correlation factor in different alloy system at different temperature",
            "ylabel": "Temperature",
            "filename": r"Vacancy correlation factor.png",
            "Normalized": False,
            "filepath": os.path.join(self.outputpath, "tempsimulationresult"),
        }

        self.comparision_temp_results_plotting(
            correlationfactor_result_df, plot_details
        )
        for i, alloysys in enumerate(correlationfactor_result_df.index):
            alloysys_result = correlationfactor_result_df.iloc[i, :]
            for k, temp in enumerate(self.temp):
                plt.scatter(
                    self.spacing * i,
                    alloysys_result[temp],
                    color=self.colors[k],
                )
        if not os.path.exists(
            os.path.join(self.outputpath, "effective barrier", "alloysystem")
        ):
            os.makedirs(
                os.path.join(self.outputpath, "effective barrier", "alloysystem")
            )
        os.chdir(os.path.join(self.outputpath, "effective barrier", "alloysystem"))
        plt.ylabel("Correlation Factor(f)")
        plt.title(f"Vacancy Correlation Factor in all alloysystem")
        xticks = [i * self.spacing for i in range(len(self.alloysystems))]
        plt.xticks(xticks)
        xticklabels = [xi for xi in self.alloysystems]
        plt.gca().set_xticks(xticks)
        plt.axvline(x=self.spacing * 4.5, color="black", linestyle="dashed")
        plt.axvline(x=self.spacing * 20.5, color="black", linestyle="dashed")
        plt.gca().set_xticklabels(xticklabels, rotation=270, ha="center")
        plt.tight_layout()
        plt.savefig(f"All alloy temp Vacancy correlation factor in {alloysys}.png")
        plt.close()
        os.chdir(self.maindir)
        return

    def vacancy_jumpingfrequency_temp_comparison(self):
        jumpingfrequency_result = {}
        for temperature, temperaturepath in zip(self.temp, self.temppath):
            jumpingfrequency_result[temperature] = {}
            for alloy in self.alloysystems:
                os.chdir(os.path.join(self.maindir, temperaturepath, alloy))
                alloy_result = np.log(self.read_jumpingfreqency_result())
                jumpingfrequency_result[temperature][alloy] = alloy_result
                os.chdir(os.path.join(self.maindir, temperaturepath))
            os.chdir(self.maindir)
        jumpingfrequency_df = pd.DataFrame.from_dict(
            jumpingfrequency_result, orient="columns"
        )

        plot_details = {
            "title": "Vacancy jumping frequency in different alloy system at different temperature",
            "ylabel": "Temperature",
            "filename": r"Vacancy jumping frequency.png",
            "Normalized": False,
            "filepath": os.path.join(self.outputpath, "tempsimulationresult"),
        }

        self.comparision_temp_results_plotting(jumpingfrequency_df, plot_details)
        return

    def vacancy_msd_temp_comparison(self):
        msd_result = {}
        for temperature, temperaturepath in zip(self.temp, self.temppath):
            msd_result[temperature] = {}
            for alloy in self.alloysystems:
                os.chdir(os.path.join(self.maindir, temperaturepath, alloy))
                alloy_result = self.read_sitevisit_sd_vacancy_result()
                msd_result[temperature][alloy] = alloy_result[3]
                os.chdir(os.path.join(self.maindir, temperaturepath))
            os.chdir(self.maindir)
        msd_result_df = pd.DataFrame.from_dict(msd_result, orient="columns")

        plot_details = {
            "title": "MSD in different alloy system at different temperature",
            "ylabel": "Temperature",
            "filename": r"MSD result.png",
            "Normalized": True,
            "filepath": os.path.join(self.outputpath, "tempsimulationresult"),
        }

        self.comparision_temp_results_plotting(msd_result_df, plot_details)
        return

    def vacancy_msd_timestep_temp(self):
        for alloy in self.alloysystems:
            for i, (temperature, temperaturpath) in enumerate(
                zip(self.temp, self.temppath)
            ):
                os.chdir(os.path.join(temperaturpath, alloy))
                record = pd.read_json("MSD_timestep.json", orient="index")
                timestep = np.array(record.index).reshape(-1, 1)
                msd = np.array(record.fillna(0))
                avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
                    timestep, msd
                )
                xfit = np.linspace(0, 100000, 1000)
                yfit = avg_model.predict(xfit[:, np.newaxis])
                plt.plot(xfit, yfit, label=temperature, color=self.colors[i])
                os.chdir(self.maindir)
            if not os.path.exists(os.path.join(self.outputpath, "TempMSDAlloysystem")):
                os.makedirs(os.path.join(self.outputpath, "TempMSDAlloysystem"))
            os.chdir(os.path.join(self.outputpath, "TempMSDAlloysystem"))
            plt.xlabel("Timestep")
            plt.ylabel("MSD")
            plt.tight_layout()
            plt.savefig(f"Temperature MSD timestep {alloy}.png")
            plt.close()
            os.chdir(self.maindir)

        return

    def vacancy_msd_time_temp(self):
        for alloy in self.alloysystems:
            for i, (temperature, temperaturpath) in enumerate(
                zip(self.temp, self.temppath)
            ):
                os.chdir(os.path.join(temperaturpath, alloy))
                msd_df = pd.read_json("MSD_real_time.json").fillna(0)
                # 進行 linear fit 數據處理
                avg_time = np.array(msd_df[f"Time_{alloy}"]).reshape(-1, 1)
                msd = np.array(msd_df[f"MSD_{alloy}"].fillna(0))
                avg_model = LinearRegression(fit_intercept=False, positive=True).fit(
                    avg_time, msd
                )
                max_time = avg_time.max()
                xfit = np.linspace(0, max_time, 1000)
                yfit = avg_model.predict(xfit[:, np.newaxis])
                plt.plot(xfit, yfit, label=temperature, color=self.colors[i])
                os.chdir(self.maindir)
            if not os.path.exists(os.path.join(self.outputpath, "TempMSDAlloysystem")):
                os.makedirs(os.path.join(self.outputpath, "TempMSDAlloysystem"))
            os.chdir(os.path.join(self.outputpath, "TempMSDAlloysystem"))
            plt.xlabel("Time")
            plt.ylabel("MSD")
            plt.tight_layout()
            plt.savefig(f"Temperature MSD time {alloy}.png")
            plt.close()
            os.chdir(self.maindir)

        return

    def validation_comparison(self):
        result = {}
        for alloy in self.alloysystems:
            result[alloy] = {}
            # 讀取 input 的值
            os.chdir(os.path.join(self.temppath[0], alloy))
            with open(f"{alloy}_Parameter.txt") as fh:
                for information in fh.readlines():
                    if "migration_energy_mu" in information:
                        energy_list = (
                            information.split(":")[1]
                            .strip("\n")
                            .replace("[", "")
                            .replace("]", "")
                            .split(" ")
                        )
                        migration_energy_list = [
                            float(energy) for energy in energy_list if energy != ""
                        ]
            result[alloy]["Input"] = np.mean(migration_energy_list)
            os.chdir(self.maindir)
            # 讀取實際fit 出來的值
            os.chdir(os.path.join(self.outputpath, "TempArrheniusAlloysystem"))
            with open(f"Diffusion coefficient of {alloy}.txt", mode="r") as fh:
                activation_energy = float(fh.readline().strip("\n").split(":")[1])
                result[alloy]["Fitted"] = activation_energy
            os.chdir(self.maindir)

        result_df = pd.DataFrame.from_dict(result, orient="index")

        return

    def run_plotting(self):
        print("Now plotting arrehenius relationship ... ")
        self.arrhenius_diffusivity()
        print("Now plotting comparison of activation energy ...")
        self.activation_energy_comparision()
        print("Now plotting comparison of d0 ...")
        self.prefactor_comparision()
        print("Now plotting all system temperature diffusivity ...")
        self.temperature_diffusivity_comparison_plotting()

    def run_temp_plotting(self):
        self.vacancy_backhop_probability_temp_comparison()
        self.vacancy_correlationfactor_temp_comparison()
        self.vacancy_diffusivity_temp_comparison()
        self.vacancy_jumpingfrequency_temp_comparison()
        self.vacancy_site_temp_comparison()
        self.vacancy_time_temp_comparison()
        self.vacancy_effctive_diffusion_barrier()
        self.vacancy_msd_timestep_temp()
        self.vacancy_msd_time_temp()
        self.validation_comparison()


