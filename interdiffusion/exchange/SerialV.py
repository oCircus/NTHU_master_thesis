import os, time, json, gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import special


class SCInterdiffusion:
    def __init__(
        self,
        crystal,
        alloy,
        elements,
        concentration,
        exchangerate,
        aspectratio=[4, 1],
        steps=100000,
        numofvacancy=100,
        repetitions=100,
        lattice_size=1000,
    ):
        self.steps = int(steps)
        self.crystal = crystal
        self.alloy = alloy
        self.numofvacancy = int(numofvacancy)
        self.repetitions = repetitions
        self.const = 0.08625  # kT
        self.exchangerate = exchangerate
        self.elements = elements
        self.lattice_size = int(lattice_size)
        self.plane_length = int(aspectratio[1]) * self.lattice_size
        self.couple_length = int(aspectratio[0]) * self.lattice_size
        self.plane_atom_num = self.plane_length**2
        self.directions = np.array(
            [
                [1, 0, 0],  # +x
                [-1, 0, 0],  # -x
                [0, 1, 0],  # +y
                [0, -1, 0],  # -y
                [0, 0, 1],  # +z
                [0, 0, -1],  # -z
            ]
        )
        self.concentration = concentration
        (
            self.upboundary,
            self.downboundary,
            self.frontboundary,
            self.backboundary,
            self.downfrontboundary,
            self.upfrontboundary,
            self.downbackboundary,
            self.upbackboundary,
            self.edgeboundary,
            self.leftrightboundary,
        ) = self.boundary_make()

    def homo_scatomlist(self):
        # Left of diffusion couple
        planar_num_of_atoms_left = [
            round(self.plane_atom_num * element_concentration)
            for element_concentration in self.concentration[0]
        ]
        planar_num_of_atoms_left[-1] = self.plane_atom_num - sum(
            planar_num_of_atoms_left[0:-1]
        )
        planar_atom_list_left = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(
                    self.elements[0], planar_num_of_atoms_left
                )
            ]
        )
        planar_num_of_atoms_right = [
            round(self.plane_atom_num * element_concentration)
            for element_concentration in self.concentration[1]
        ]
        planar_num_of_atoms_right[-1] = self.plane_atom_num - sum(
            planar_num_of_atoms_right[0:-1]
        )
        planar_atom_list_right = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(
                    self.elements[1], planar_num_of_atoms_right
                )
            ]
        )
        return planar_atom_list_left, planar_atom_list_right

    def homo_scdiffusioncouple(self, planar_atom_list_left, planar_atom_list_right):
        # Left diffusion coouple
        left_couple = np.array([])
        for _ in range(self.couple_length):
            planer_atom = planar_atom_list_left
            np.random.shuffle(planer_atom)
            left_couple = np.concatenate(([left_couple, planer_atom]))

        right_couple = np.array([])
        for _ in range(self.couple_length):
            planer_atom = planar_atom_list_right
            np.random.shuffle(planer_atom)
            right_couple = np.concatenate(([right_couple, planer_atom]))
        lattice = np.concatenate([left_couple, right_couple])
        return lattice.astype(int)

    def boundary_make(self):
        couple_total_index = 2 * self.couple_length * (self.plane_atom_num)

        left_boundary = np.arange(self.plane_atom_num)
        up_boundary = np.arange(
            self.plane_length - 1, couple_total_index, self.plane_length
        )
        front_boundary = np.concatenate(
            [
                np.arange(i, i + self.plane_length)
                for i in range(0, couple_total_index, self.plane_atom_num)
            ]
        )
        back_boundary = np.concatenate(
            [
                np.arange(i, i + self.plane_length)
                for i in range(
                    self.plane_atom_num - self.plane_length,
                    couple_total_index,
                    self.plane_atom_num,
                )
            ]
        )
        down_boundary = np.arange(0, couple_total_index, self.plane_length)
        right_boundary = np.arange(
            couple_total_index - self.plane_atom_num, couple_total_index
        )

        # Edge Boundary
        down_front_boundary = np.arange(0, couple_total_index, self.plane_atom_num)
        up_front_boundary = np.arange(
            self.plane_length - 1, couple_total_index, self.plane_length**2
        )
        down_back_boundary = np.arange(
            self.plane_atom_num - self.plane_length,
            couple_total_index,
            self.plane_atom_num,
        )
        up_back_boundary = np.arange(
            self.plane_atom_num - 1, couple_total_index, self.plane_length**2
        )

        edge_boundary = np.unique(
            np.concatenate([up_boundary, down_boundary, front_boundary, back_boundary])
        )
        left_right_boundary = np.unique(np.concatenate([left_boundary, right_boundary]))

        return (
            up_boundary,
            down_boundary,
            front_boundary,
            back_boundary,
            down_front_boundary,
            up_front_boundary,
            down_back_boundary,
            up_back_boundary,
            edge_boundary,
            left_right_boundary,
        )

    def check_bound(self, position_index):
        return np.any(np.isin(position_index, self.leftrightboundary))

    def calc_concentration(self, lattice):
        composition_record = {}
        for i, x in enumerate(
            range(0, 2 * self.couple_length * self.plane_atom_num, self.plane_atom_num)
        ):
            atoms = lattice[x : x + 1 * self.plane_atom_num]
            element, concentration = np.unique(atoms, return_counts=True)
            composition_record[i] = dict(zip(element, concentration / len(atoms)))
        concentration_df = pd.DataFrame.from_dict(composition_record, orient="index")
        concentration_df = concentration_df.fillna(0)

        return concentration_df

    def find_gradient_layer(
        self, left_idz_layer, right_idz_layer, concentration_profile, bear_layer
    ):
        left_before_con = concentration_profile.iloc[left_idz_layer, 0]
        left_after_con = concentration_profile.iloc[left_idz_layer - bear_layer, 0]
        right_before_con = concentration_profile.iloc[right_idz_layer, 0]
        right_after_con = concentration_profile.iloc[right_idz_layer + bear_layer, 0]

        if left_after_con - left_before_con >= 0.02:
            left_idz_layer -= 1
        if right_before_con - right_after_con >= 0.02:
            right_idz_layer += 1

        return left_idz_layer, right_idz_layer

    def idzboundary_index(self, left_idz_layer, right_idz_layer):
        return np.arange(
            self.plane_atom_num * self.couple_length
            + ((left_idz_layer - self.couple_length) - 1) * self.plane_atom_num,
            self.plane_atom_num * self.couple_length
            + (right_idz_layer - self.couple_length) * self.plane_atom_num,
        )

    def check_in_gradient_layer(self, position_index, idzboundary):
        return np.any(np.isin(position_index, idzboundary))

    def vacancy_create(self, lattice):
        """
        Create a vacancy in center of lattice site, and return both new lattice and index of vancancy position
        """
        center = int(
            self.couple_length * self.plane_atom_num - (self.plane_atom_num // 2)
        )
        remove_atom_type = lattice[center]
        lattice[center] = 0
        return lattice, center, remove_atom_type

    def v_position_layer(self, v_position):
        return v_position // self.plane_atom_num

    def find_neighbor(self, lattice, v_position):
        """
        Find nearest positoin of  +x,-x,+y,-y,+z,-z direction
        """
        if np.any(np.isin(v_position, self.edgeboundary)):
            if np.any(np.isin(v_position, self.downfrontboundary)):
                # print("Reach DownFrontboundary")
                neighbor_index = [
                    v_position + self.plane_length**2,
                    v_position - self.plane_length**2,
                    v_position + self.plane_length,
                    v_position - self.plane_length + self.plane_length**2,
                    v_position + 1,
                    v_position - 1 + self.plane_length,
                ]
            elif np.any(np.isin(v_position, self.downbackboundary)):
                # print("Reach DownBackboundary")
                neighbor_index = [
                    v_position + self.plane_length**2,
                    v_position - self.plane_length**2,
                    v_position + self.plane_length - self.plane_length**2,
                    v_position - self.plane_length,
                    v_position + 1,
                    v_position - 1 + self.plane_length,
                ]
            elif np.any(np.isin(v_position, self.upfrontboundary)):
                # print("Reach UpFrontboundary")
                neighbor_index = [
                    v_position + self.plane_length**2,
                    v_position - self.plane_length**2,
                    v_position + self.plane_length,
                    v_position - self.plane_length + self.plane_length**2,
                    v_position + 1 - self.plane_length,
                    v_position - 1,
                ]
            elif np.any(np.isin(v_position, self.upbackboundary)):
                # print("Reach UpBackboundary")
                neighbor_index = [
                    v_position + self.plane_length**2,
                    v_position - self.plane_length**2,
                    v_position + self.plane_length - self.plane_length**2,
                    v_position - self.plane_length,
                    v_position + 1 - self.plane_length,
                    v_position - 1,
                ]
            elif np.any(np.isin(v_position, self.upboundary)):
                # print("Reach UPboundary")
                neighbor_index = [
                    v_position + self.plane_length**2,
                    v_position - self.plane_length**2,
                    v_position + self.plane_length,
                    v_position - self.plane_length,
                    v_position + 1 - self.plane_length,
                    v_position - 1,
                ]
            elif np.any(np.isin(v_position, self.downboundary)):
                # print("Reach DOWNboundary")
                neighbor_index = [
                    v_position + self.plane_length**2,
                    v_position - self.plane_length**2,
                    v_position + self.plane_length,
                    v_position - self.plane_length,
                    v_position + 1,
                    v_position - 1 + self.plane_length,
                ]
            elif np.any(np.isin(v_position, self.frontboundary)):
                # print("Reach FRONTboundary")
                neighbor_index = [
                    v_position + self.plane_length**2,
                    v_position - self.plane_length**2,
                    v_position + self.plane_length,
                    v_position - self.plane_length + self.plane_length**2,
                    v_position + 1,
                    v_position - 1,
                ]
            elif np.any(np.isin(v_position, self.backboundary)):
                # print("Reach BACKboundary")
                neighbor_index = [
                    v_position + self.plane_length**2,
                    v_position - self.plane_length**2,
                    v_position + self.plane_length - self.plane_length**2,
                    v_position - self.plane_length,
                    v_position + 1,
                    v_position - 1,
                ]
        else:
            neighbor_index = [
                v_position + self.lattice_size**2,
                v_position - self.lattice_size**2,
                v_position + self.lattice_size,
                v_position - self.lattice_size,
                v_position + 1,
                v_position - 1,
            ]
        neighbor = np.array([lattice[indexing] for indexing in neighbor_index])
        return neighbor_index, neighbor

    def prob(self, neighbor_atom):
        ex_probability = np.array(
            [self.exchangerate[int(atom) - 1] for atom in neighbor_atom]
        )
        nor_ex_probability = ex_probability / sum(ex_probability)

        return nor_ex_probability

    def determine_exchange(self, neighbor_index, exchange_prob):
        """
        Determine the exchange direction
        """
        chosen_direction = np.random.choice(self.directions.shape[0], p=exchange_prob)
        chosen_index = neighbor_index[chosen_direction]
        chosen_vector = self.directions[chosen_direction]

        return chosen_index, chosen_vector

    def perform_exchnage(self, lattice, v_position, change_direction_index):
        """
        Perform exchange
        """
        lattice[v_position], lattice[change_direction_index] = (
            lattice[change_direction_index],
            lattice[v_position],
        )
        v_position = change_direction_index
        return lattice, v_position

    def record_exchange(self, record, step, direction):
        record[step] = direction[0:3]
        return record

    def rep_lattice_record(self, lattice, v_position, remove_atom_type):
        lattice[v_position] = remove_atom_type
        return lattice

    def random_walk(self):
        walkers_record = {}  # 紀錄vacancy移動軌跡
        lattice_record = {}  # 紀錄lattice
        v_position_record = {}  # 紀錄空孔最終位置
        idz_record = {}
        num_repetitions = 0
        while num_repetitions <= self.repetitions:
            # 進行重複取樣
            walkers_record[num_repetitions] = {}
            lattice_record[num_repetitions] = {}
            v_position_record[num_repetitions] = {}
            idz_recording = np.empty((self.numofvacancy + 1, 2), dtype=int)
            atom_list_left, atom_list_right = self.homo_scatomlist()
            sc_lattice = self.homo_scdiffusioncouple(atom_list_left, atom_list_right)
            left_idz_layer, right_idz_layer = (
                self.couple_length - 1,
                self.couple_length + 1,
            )
            success = 0
            while success <= self.numofvacancy:
                # 再擺放前，先計算flux profile，再根據profile，找出idz range
                flux_result = self.calc_concentration(sc_lattice)
                left_idz_layer, right_idz_layer = self.find_gradient_layer(
                    left_idz_layer, right_idz_layer, flux_result, 1
                )
                idz_recording[success][0], idz_recording[success][1] = (
                    left_idz_layer,
                    right_idz_layer,
                )
                idz_idnex = self.idzboundary_index(left_idz_layer, right_idz_layer)
                print(
                    f"Now in {self.alloy} {num_repetitions} {success} with {left_idz_layer, right_idz_layer}"
                )
                # Lattice construct and randomly place atom onto it.
                lattice, v_position_index, remove_atom_type = self.vacancy_create(
                    sc_lattice
                )
                vacancy_record = np.empty((self.steps, 3), dtype=float)
                position_recording = np.empty((self.steps,), dtype=int)  # 紀錄空孔位置
                try:
                    for step in range(self.steps):
                        if self.check_in_gradient_layer(v_position_index, idz_idnex):
                            neighbor_index, neighbor = self.find_neighbor(
                                lattice, v_position_index
                            )
                            # Determine the exchange diretion
                            exchange_probability = self.prob(neighbor)
                            (
                                v_after_index,
                                v_movement_vector,
                            ) = self.determine_exchange(
                                neighbor_index, exchange_probability
                            )
                            # Perform the vacancy exchangement with chosen atom and direciton
                            lattice, v_position_index = self.perform_exchnage(
                                lattice, v_position_index, v_after_index
                            )

                            # Record the exchange moment
                            vacancy_record = self.record_exchange(
                                vacancy_record, step, v_movement_vector
                            )
                            position_recording[step] = v_position_index
                        else:
                            vacancy_record = vacancy_record[0:step]
                            position_recording = position_recording[0:step]
                            break

                except Exception as e:
                    print(e)
                    if success in lattice_record[num_repetitions]:
                        del lattice_record[num_repetitions][success]
                    continue

                vacancy_trace = np.cumsum(vacancy_record, axis=0)
                walkers_record[num_repetitions]["x" + str(success)] = vacancy_trace[
                    :, 0
                ]
                walkers_record[num_repetitions]["y" + str(success)] = vacancy_trace[
                    :, 1
                ]
                walkers_record[num_repetitions]["z" + str(success)] = vacancy_trace[
                    :, 2
                ]
                walkers_record[num_repetitions]["SD" + str(success)] = np.sum(
                    np.square(vacancy_trace), axis=1
                )
                v_position_record[num_repetitions][success] = position_recording

                sc_lattice = self.rep_lattice_record(
                    lattice, v_position_index, remove_atom_type
                )
                if success % 10 == 0:
                    lattice_record[num_repetitions][success] = np.copy(sc_lattice)
                success += 1
            idz_record[num_repetitions] = idz_recording.reshape(-1)
            num_repetitions += 1
        vacancy_trace = pd.DataFrame(walkers_record)
        filtered_lattice_record = pd.DataFrame(lattice_record)
        v_position = pd.DataFrame(v_position_record)
        idz = pd.DataFrame(idz_record)

        return vacancy_trace, filtered_lattice_record, v_position, idz

    def save_data(
        self, parameter, vacancy_record, lattice_record, v_position_record, idz
    ):
        """
        Save the data with DataFrame in form of json
        """
        currentdir = os.path.abspath(os.curdir)
        dirname = f"sc_serial_dynamicIDZ_diffpro_data_size{parameter[-1]}_numofvacancy{parameter[-3]}_{parameter[1]}"
        # file_dir = r"/work1/u3284480/"
        dirpath = os.path.join(currentdir, "serial", "0.02", f"{self.crystal}", dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        os.chdir(dirpath)
        data_path = os.path.join(dirpath, f"{self.alloy}")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        os.chdir(data_path)
        lattice_record.to_json(
            f"Lattice record of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        vacancy_record.to_json(
            f"Vacancy trace of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        v_position_record.to_json(
            f"Vacancy position of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        idz.to_json(
            f"idz of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        with open(f"{self.alloy}_Parameter.txt", "w") as fh:
            parameter_name = [
                "alloy",
                "fluc",
                "elements",
                "concentration",
                "exchangerate",
                "ratio",
                "steps",
                "numofvacancy",
                "repetitions",
                "lattice_size",
            ]
            for name, para in zip(parameter_name, parameter):
                fh.write(f"{name}:{para}\n")
        os.chdir(currentdir)

        return


class FCCInterdiffusion:
    def __init__(
        self,
        crystal,
        alloy,
        elements,
        concentration,
        exchangerate,
        aspectratio=[4, 1],
        steps=100000,
        numofvacancy=100,
        repetitions=100,
        lattice_size=1000,
    ):
        self.steps = int(steps)
        self.crystal = crystal
        self.alloy = alloy
        self.numofvacancy = int(numofvacancy)
        self.repetitions = repetitions
        self.const = 0.08625  # kT
        self.exchangerate = exchangerate
        self.elements = elements
        self.lattice_size = int(lattice_size)
        self.plane_length = int(aspectratio[1]) * self.lattice_size
        self.couple_length = int(aspectratio[0]) * self.lattice_size
        self.plane_atom_num = self.plane_length**2
        self.directions = np.array(
            [
                [0.5, 0.5, 0],  # UPPER SHELL and up-right
                [0, 0.5, 0.5],  # UPPER SHELL and up-left
                [0.5, 0, 0.5],  # UpPER SHELL and down
                [0.5, 0, -0.5],  # Right
                [0, 0.5, -0.5],  # up-right
                [-0.5, 0.5, 0],  # up-left
                [-0.5, 0, 0.5],  # left
                [0, -0.5, 0.5],  # down-left
                [0.5, -0.5, 0],  # down-right
                [-0.5, 0, -0.5],  # Down shell and up
                [-0.5, -0.5, 0],  # Down shell and left
                [0, -0.5, -0.5],  # Down shell and right
            ]
        )
        self.concentration = concentration
        (
            self.upboundary,
            self.downboundary,
            self.frontboundary,
            self.backboundary,
            self.downfrontboundary,
            self.upfrontboundary,
            self.downbackboundary,
            self.upbackboundary,
            self.edgeboundary,
            self.leftrightboundary,
        ) = self.boundary_make()

    def homo_fccatomlist(self):
        # Left of diffusion couple
        planar_num_of_atoms_left = [
            round(self.plane_atom_num * element_concentration)
            for element_concentration in self.concentration[0]
        ]
        planar_num_of_atoms_left[-1] = self.plane_atom_num - sum(
            planar_num_of_atoms_left[0:-1]
        )
        planar_atom_list_left = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(
                    self.elements[0], planar_num_of_atoms_left
                )
            ]
        )
        planar_num_of_atoms_right = [
            round(self.plane_atom_num * element_concentration)
            for element_concentration in self.concentration[1]
        ]
        planar_num_of_atoms_right[-1] = self.plane_atom_num - sum(
            planar_num_of_atoms_right[0:-1]
        )
        planar_atom_list_right = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(
                    self.elements[1], planar_num_of_atoms_right
                )
            ]
        )
        return planar_atom_list_left, planar_atom_list_right

    def homo_fccdiffusioncouple(self, planar_atom_list_left, planar_atom_list_right):
        # Left diffusion coouple
        left_couple = np.array([])
        for _ in range(self.couple_length):
            planer_atom = planar_atom_list_left
            np.random.shuffle(planer_atom)
            left_couple = np.concatenate(([left_couple, planer_atom]))

        right_couple = np.array([])
        for _ in range(self.couple_length):
            planer_atom = planar_atom_list_right
            np.random.shuffle(planer_atom)
            right_couple = np.concatenate(([right_couple, planer_atom]))
        lattice = np.concatenate([left_couple, right_couple])
        return lattice.astype(int)

    def boundary_make(self):
        couple_total_index = 2 * self.couple_length * (self.plane_atom_num)

        left_boundary = np.arange(self.plane_atom_num)
        up_boundary = np.arange(
            self.plane_length - 1, couple_total_index, self.plane_length
        )
        front_boundary = np.concatenate(
            [
                np.arange(i, i + self.plane_length)
                for i in range(0, couple_total_index, self.plane_atom_num)
            ]
        )
        back_boundary = np.concatenate(
            [
                np.arange(i, i + self.plane_length)
                for i in range(
                    self.plane_atom_num - self.plane_length,
                    couple_total_index,
                    self.plane_atom_num,
                )
            ]
        )
        down_boundary = np.arange(0, couple_total_index, self.plane_length)
        right_boundary = np.arange(
            couple_total_index - self.plane_atom_num, couple_total_index
        )

        # Edge Boundary
        down_front_boundary = np.arange(0, couple_total_index, self.plane_atom_num)
        up_front_boundary = np.arange(
            self.plane_length - 1, couple_total_index, self.plane_length**2
        )
        down_back_boundary = np.arange(
            self.plane_atom_num - self.plane_length,
            couple_total_index,
            self.plane_atom_num,
        )
        up_back_boundary = np.arange(
            self.plane_atom_num - 1, couple_total_index, self.plane_length**2
        )

        edge_boundary = np.unique(
            np.concatenate([up_boundary, down_boundary, front_boundary, back_boundary])
        )
        left_right_boundary = np.unique(np.concatenate([left_boundary, right_boundary]))
        return (
            up_boundary,
            down_boundary,
            front_boundary,
            back_boundary,
            down_front_boundary,
            up_front_boundary,
            down_back_boundary,
            up_back_boundary,
            edge_boundary,
            left_right_boundary,
        )

    def check_bound(self, position_index):
        return np.any(np.isin(position_index, self.leftrightboundary))

    def calc_concentration(self, lattice):
        composition_record = {}
        for i, x in enumerate(
            range(0, 2 * self.couple_length * self.plane_atom_num, self.plane_atom_num)
        ):
            atoms = lattice[x : x + 1 * self.plane_atom_num]
            element, concentration = np.unique(atoms, return_counts=True)
            composition_record[i] = dict(zip(element, concentration / len(atoms)))
        concentration_df = pd.DataFrame.from_dict(composition_record, orient="index")
        concentration_df = concentration_df.fillna(0)

        return concentration_df

    def find_gradient_layer(
        self, left_idz_layer, right_idz_layer, concentration_profile, bear_layer
    ):
        left_before_con = concentration_profile.iloc[left_idz_layer, 0]
        left_after_con = concentration_profile.iloc[left_idz_layer - bear_layer, 0]
        right_before_con = concentration_profile.iloc[right_idz_layer, 0]
        right_after_con = concentration_profile.iloc[right_idz_layer + bear_layer, 0]

        if left_after_con - left_before_con >= 0.02:
            left_idz_layer -= 1
        if right_before_con - right_after_con >= 0.02:
            right_idz_layer += 1

        return left_idz_layer, right_idz_layer

    def idzboundary_index(self, left_idz_layer, right_idz_layer):
        return np.arange(
            self.plane_atom_num * self.couple_length
            + ((left_idz_layer - self.couple_length) - 1) * self.plane_atom_num,
            self.plane_atom_num * self.couple_length
            + (right_idz_layer - self.couple_length) * self.plane_atom_num,
        )

    def check_in_gradient_layer(self, position_index, idzboundary):
        return np.any(np.isin(position_index, idzboundary))

    def vacancy_create(self, lattice):
        """
        Create a vacancy in center of lattice site, and return both new lattice and index of vancancy position
        """
        center = int(
            self.couple_length * self.plane_atom_num - (self.plane_atom_num // 2)
        )
        remove_atom_type = lattice[center]
        lattice[center] = 0
        return lattice, center, remove_atom_type

    def v_position_layer(self, v_position):
        return v_position // self.plane_atom_num

    def find_neighbor_adjust(self, v_position_index, adjust):
        neighbor_index = []
        for a in adjust:
            neighbor_index.append(
                v_position_index
                + self.plane_atom_num * a[0]
                + self.plane_length * a[1]
                + a[2]
            )

        return neighbor_index

    def find_neighbor(self, lattice, v_position):
        locationlayer = self.v_position_layer(v_position)
        row_index = (
            v_position - locationlayer * self.plane_atom_num
        ) // self.plane_length

        if np.any(np.isin(v_position, self.edgeboundary)):
            boundary_type = [
                "upbackboundary",
                "downbackboundary",
                "upfrontboundary",
                "downfrontboundary",
                "downboundary",
                "upboundary",
                "frontboundary",
                "backboundary",
            ]
            boundaries = [
                self.upbackboundary,
                self.downbackboundary,
                self.upfrontboundary,
                self.downfrontboundary,
                self.downboundary,
                self.upboundary,
                self.frontboundary,
                self.backboundary,
            ]
            boundary_info = {
                "upbackboundary": {
                    1: [
                        [
                            [-1, -1, 1],
                            [-1, 0, 0],
                            [-1, -2, 1],
                            [0, -1, 1],
                            [-1, 0, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -2, 1],
                            [1, 0, 0],
                            [1, -1, 0],
                            [1, -2, 1],
                        ],
                        [
                            [-1, -1, 1],
                            [-1, 0, 0],
                            [-1, -2, 1],
                            [0, -1, 1],
                            [-1, 0, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -2, 1],
                            [1, 0, 0],
                            [1, -1, 0],
                            [1, -2, 1],
                        ],
                    ],
                    0: [
                        [
                            [-1, 0, 0],
                            [-1, 0, -1],
                            [-1, -1, 0],
                            [0, -1, 1],
                            [-1, 0, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -2, 1],
                            [0, 1, 0],
                            [1, 0, -1],
                            [1, 0, 0],
                        ],
                        [
                            [-1, 0, 0],
                            [-1, 0, -1],
                            [-1, -1, 0],
                            [0, -1, 1],
                            [-1, 0, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -2, 1],
                            [0, 1, 0],
                            [1, 0, -1],
                            [1, 0, 0],
                        ],
                    ],
                    2: [
                        [
                            [-2, 0, 1],
                            [-2, 1, 0],
                            [-1, 0, 0],
                            [0, -1, 1],
                            [-1, 0, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -2, 1],
                            [0, 0, 1],
                            [1, 0, 0],
                            [1, -1, 1],
                        ],
                        [
                            [-2, 0, 1],
                            [-2, 1, 0],
                            [-1, 0, 0],
                            [0, -1, 1],
                            [-1, 0, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -2, 1],
                            [0, 0, 1],
                            [1, 0, 0],
                            [1, -1, 1],
                        ],
                    ],
                },
                "downbackboundary": {
                    1: [
                        [
                            [-1, 0, 1],
                            [-1, 0, 0],
                            [-1, -1, 1],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 1, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [1, 0, 0],
                            [1, -1, 0],
                            [1, -1, 1],
                        ],
                        [
                            [-1, 0, 1],
                            [-1, 0, 0],
                            [-1, -1, 1],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 1, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [1, 0, 0],
                            [1, -1, 0],
                            [1, -1, 1],
                        ],
                    ],
                    0: [
                        [
                            [-1, 0, 0],
                            [-1, 1, -1],
                            [-1, -1, 0],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 1, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [0, 1, 0],
                            [1, 1, -1],
                            [1, 0, 0],
                        ],
                        [
                            [-1, 0, 0],
                            [-1, 1, -1],
                            [-1, -1, 0],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 1, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [0, 1, 0],
                            [1, 1, -1],
                            [1, 0, 0],
                        ],
                    ],
                    2: [
                        [
                            [-2, 1, 1],
                            [-2, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 1, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [0, 1, 1],
                            [1, 0, 0],
                            [1, 0, 1],
                        ],
                        [
                            [-2, 1, 1],
                            [-2, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 1, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [0, 1, 1],
                            [1, 0, 0],
                            [1, 0, 1],
                        ],
                    ],
                },
                "upfrontboundary": {
                    1: [
                        [
                            [-1, -1, 1],
                            [-1, 0, 0],
                            [0, -1, 0],
                            [0, -1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 0, 0],
                            [2, -1, -1],
                            [2, -1, 0],
                        ],
                        [
                            [-1, -1, 1],
                            [-1, 0, 0],
                            [0, -1, 0],
                            [0, -1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 0, 0],
                            [2, -1, -1],
                            [2, -1, 0],
                        ],
                    ],
                    0: [
                        [
                            [-1, 0, 0],
                            [-1, 0, -1],
                            [0, -1, -1],
                            [0, -1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 1, -1],
                            [1, 0, -1],
                            [1, 0, 0],
                        ],
                        [
                            [-1, 0, 0],
                            [-1, 0, -1],
                            [0, -1, -1],
                            [0, -1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 1, -1],
                            [1, 0, -1],
                            [1, 0, 0],
                        ],
                    ],
                    2: [
                        [
                            [-1, 1, 0],
                            [-1, 1, -1],
                            [-1, 0, 0],
                            [0, -1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 1, 0],
                            [1, 0, 0],
                            [1, -1, 1],
                        ],
                        [
                            [-1, 1, 0],
                            [-1, 1, -1],
                            [-1, 0, 0],
                            [0, -1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 1, 0],
                            [1, 0, 0],
                            [1, -1, 1],
                        ],
                    ],
                },
                "downfrontboundary": {
                    1: [
                        [
                            [-1, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 2, -1],
                            [0, 1, -1],
                            [1, 0, -1],
                            [1, -1, 0],
                            [1, 0, 0],
                            [2, 0, -1],
                            [2, -1, 0],
                        ],
                        [
                            [-1, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 2, -1],
                            [0, 1, -1],
                            [1, 0, -1],
                            [1, -1, 0],
                            [1, 0, 0],
                            [2, 0, -1],
                            [2, -1, 0],
                        ],
                    ],
                    0: [
                        [
                            [-1, 0, 0],
                            [-1, 1, -1],
                            [0, 0, -1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 2, -1],
                            [0, 1, -1],
                            [1, 0, -1],
                            [1, -1, 0],
                            [1, 2, -1],
                            [1, 1, -1],
                            [1, 0, 0],
                        ],
                        [
                            [-1, 0, 0],
                            [-1, 1, -1],
                            [0, 0, -1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 2, -1],
                            [0, 1, -1],
                            [1, 0, -1],
                            [1, -1, 0],
                            [1, 2, -1],
                            [1, 1, -1],
                            [1, 0, 0],
                        ],
                    ],
                    2: [
                        [
                            [-1, 1, 0],
                            [-1, 2, -1],
                            [-1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 2, -1],
                            [0, 1, -1],
                            [1, 0, -1],
                            [1, -1, 0],
                            [1, 1, 0],
                            [1, 0, 0],
                            [1, 0, 1],
                        ],
                        [
                            [-1, 1, 0],
                            [-1, 2, -1],
                            [-1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 2, -1],
                            [0, 1, -1],
                            [1, 0, -1],
                            [1, -1, 0],
                            [1, 1, 0],
                            [1, 0, 0],
                            [1, 0, 1],
                        ],
                    ],
                },
                "downboundary": {
                    1: [
                        [
                            [-1, 0, 1],
                            [-1, 0, 0],
                            [-1, -1, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 2, -1],
                            [0, 1, -1],
                            [0, 0, -1],
                            [0, -1, 0],
                            [1, 0, 0],
                            [1, 0, -1],
                            [1, -1, 0],
                        ],  # row_index is even
                        [
                            [-1, 0, 1],
                            [-1, 0, 0],
                            [-1, -1, 1],  # row_index is odd
                            [0, 0, 1],
                            [0, 1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [1, 0, 0],
                            [1, -1, 0],
                            [1, -1, 1],
                        ],
                    ],
                    0: [
                        [
                            [-1, 0, 0],
                            [-1, 1, -1],
                            [-1, 0, -1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 2, -1],
                            [0, 1, -1],
                            [0, 0, -1],
                            [0, -1, 0],
                            [1, 2, -1],
                            [1, 1, -1],
                            [1, 0, 0],
                        ],
                        [
                            [-1, 0, 0],
                            [-1, 1, -1],
                            [-1, -1, 0],
                            [0, 0, 1],
                            [0, 1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [1, 1, 0],
                            [1, 1, -1],
                            [1, 0, 0],
                        ],
                    ],
                    2: [
                        [
                            [-1, 1, 0],
                            [-1, 2, -1],
                            [-1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 2, -1],
                            [0, 1, -1],
                            [0, 0, -1],
                            [0, -1, 0],
                            [1, 1, 0],
                            [1, 0, 0],
                            [1, 0, 1],
                        ],
                        [
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [1, 1, 1],
                            [1, 0, 0],
                            [1, 0, 1],
                        ],
                    ],
                },
                "upboundary": {
                    1: [
                        [
                            [-1, -1, 1],
                            [-1, 0, 0],
                            [-1, -1, 0],
                            [0, -1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [0, -1, -1],
                            [0, -1, 0],
                            [1, 0, 0],
                            [1, -1, -1],
                            [1, -1, 0],
                        ],  # row_index is even
                        [
                            [-1, -1, 1],
                            [-1, 0, 0],
                            [-1, -2, 1],  # row_index is odd
                            [0, -1, 1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -2, 1],
                            [1, 0, 0],
                            [1, -1, 0],
                            [1, -2, 1],
                        ],
                    ],
                    0: [
                        [
                            [-1, 0, 0],
                            [-1, 0, -1],
                            [-1, -1, -1],
                            [0, -1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [0, -1, -1],
                            [0, -1, 0],
                            [1, 1, -1],
                            [1, 0, -1],
                            [1, 0, 0],
                        ],
                        [
                            [-1, 0, 0],
                            [-1, 0, -1],
                            [-1, -1, 0],
                            [0, -1, 1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -2, 1],
                            [1, 1, 0],
                            [1, 0, -1],
                            [1, 0, 0],
                        ],
                    ],
                    2: [
                        [
                            [-1, 1, 0],
                            [-1, 1, -1],
                            [-1, 0, 0],
                            [0, -1, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [0, -1, -1],
                            [0, -1, 0],
                            [1, 1, 0],
                            [1, 0, 0],
                            [1, -1, 1],
                        ],
                        [
                            [-1, 0, 1],
                            [-1, 1, 0],
                            [-1, 0, 0],
                            [0, -1, 1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -2, 1],
                            [1, 0, 1],
                            [1, 0, 0],
                            [1, -1, 1],
                        ],
                    ],
                },
                "frontboundary": {
                    1: [
                        [
                            [-1, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 0, 0],
                            [2, -1, -1],
                            [2, -1, 0],
                        ],  # row_index is even
                        [
                            [-1, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 0, 0],
                            [2, -1, -1],
                            [2, -1, 0],
                        ],
                    ],
                    0: [
                        [
                            [-1, 0, 0],
                            [-1, 0, -1],
                            [0, -1, -1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 1, -1],
                            [1, 0, -1],
                            [1, 0, 0],
                        ],
                        [
                            [-1, 0, 0],
                            [-1, 0, -1],
                            [0, -1, -1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 1, -1],
                            [1, 0, -1],
                            [1, 0, 0],
                        ],
                    ],
                    2: [
                        [
                            [-1, 1, 0],
                            [-1, 1, -1],
                            [-1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 1, 0],
                            [1, 0, 0],
                            [1, 0, 1],
                        ],
                        [
                            [-1, 1, 0],
                            [-1, 1, -1],
                            [-1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 1, -1],
                            [0, 0, -1],
                            [1, -1, -1],
                            [1, -1, 0],
                            [1, 1, 0],
                            [1, 0, 0],
                            [1, 0, 1],
                        ],
                    ],
                },
                "backboundary": {
                    1: [
                        [
                            [-1, 0, 1],
                            [-1, 0, 0],
                            [-1, -1, 1],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [1, 0, 0],
                            [1, -1, 0],
                            [1, -1, 1],
                        ],  # row_index is even
                        [
                            [-1, 0, 1],
                            [-1, 0, 0],
                            [-1, -1, 1],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [1, 0, 0],
                            [1, -1, 0],
                            [1, -1, 1],
                        ],
                    ],
                    0: [
                        [
                            [-1, 0, 0],
                            [-1, 0, -1],
                            [-1, -1, 0],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [0, 1, 0],
                            [1, 0, -1],
                            [1, 0, 0],
                        ],
                        [
                            [-1, 0, 0],
                            [-1, 0, -1],
                            [-1, -1, 0],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [0, 1, 0],
                            [1, 0, -1],
                            [1, 0, 0],
                        ],
                    ],
                    2: [
                        [
                            [-2, 1, 1],
                            [-2, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [0, 1, 1],
                            [1, 0, 0],
                            [1, 0, 1],
                        ],
                        [
                            [-2, 1, 1],
                            [-2, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1],
                            [-1, 1, 1],
                            [-1, 1, 0],
                            [0, 0, -1],
                            [0, -1, 0],
                            [0, -1, 1],
                            [0, 1, 1],
                            [1, 0, 0],
                            [1, 0, 1],
                        ],
                    ],
                },
            }

            for boundary, boundary_index in zip(boundary_type, boundaries):
                if np.any(np.isin(v_position, boundary_index)):
                    neighbor_index = self.find_neighbor_adjust(
                        v_position,
                        boundary_info[boundary][locationlayer % 3][row_index % 2],
                    )
                    neighbor_atom = [lattice[index] for index in neighbor_index]
                    return neighbor_index, neighbor_atom

        else:
            without_bound_adjust = {
                1: [
                    [
                        [-1, 0, 1],
                        [-1, 0, 0],
                        [-1, -1, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, -1],
                        [0, 0, -1],
                        [0, -1, -1],
                        [0, -1, 0],
                        [1, 0, 0],
                        [1, -1, -1],
                        [1, -1, 0],
                    ],
                    [
                        [-1, 0, 1],
                        [-1, 0, 0],
                        [-1, -1, 1],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [0, 0, -1],
                        [0, -1, 0],
                        [0, -1, 1],
                        [1, 0, 0],
                        [1, -1, 0],
                        [1, -1, 1],
                    ],
                ],
                0: [
                    [
                        [-1, 0, 0],
                        [-1, 0, -1],
                        [-1, -1, -1],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, -1],
                        [0, 0, -1],
                        [0, -1, -1],
                        [0, -1, 0],
                        [1, 1, -1],
                        [1, 0, -1],
                        [1, 0, 0],
                    ],
                    [
                        [-1, 0, 0],
                        [-1, 0, -1],
                        [-1, -1, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [0, 0, -1],
                        [0, -1, 0],
                        [0, -1, 1],
                        [1, 1, 0],
                        [1, 0, -1],
                        [1, 0, 0],
                    ],
                ],
                2: [
                    [
                        [-1, 1, 0],
                        [-1, 1, -1],
                        [-1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, -1],
                        [0, 0, -1],
                        [0, -1, -1],
                        [0, -1, 0],
                        [1, 1, 0],
                        [1, 0, 0],
                        [1, 0, 1],
                    ],
                    [
                        [-1, 1, 1],
                        [-1, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [0, 0, -1],
                        [0, -1, 0],
                        [0, -1, 1],
                        [1, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                    ],
                ],
            }
            neighbor_index = self.find_neighbor_adjust(
                v_position, without_bound_adjust[locationlayer % 3][row_index % 2]
            )
            neighbor_atom = [lattice[index] for index in neighbor_index]
        return neighbor_index, neighbor_atom

    def prob(self, neighbor_atom):
        ex_probability = np.array(
            [self.exchangerate[int(atom) - 1] for atom in neighbor_atom]
        )
        nor_ex_probability = ex_probability / sum(ex_probability)

        return nor_ex_probability

    def determine_exchange(self, neighbor_index, exchange_prob):
        """
        Determine the exchange direction
        """
        chosen_direction = np.random.choice(self.directions.shape[0], p=exchange_prob)
        chosen_index = neighbor_index[chosen_direction]
        chosen_vector = self.directions[chosen_direction]

        return chosen_index, chosen_vector

    def perform_exchnage(self, lattice, v_position, change_direction_index):
        """
        Perform exchange
        """
        lattice[v_position], lattice[change_direction_index] = (
            lattice[change_direction_index],
            lattice[v_position],
        )
        v_position = change_direction_index
        return lattice, v_position

    def record_exchange(self, record, step, direction):
        record[step] = direction[0:3]
        return record

    def rep_lattice_record(self, lattice, v_position, remove_atom_type):
        lattice[v_position] = remove_atom_type
        return lattice

    def random_walk(self):
        walkers_record = {}  # 紀錄vacancy移動軌跡
        lattice_record = {}  # 紀錄lattice
        v_position_record = {}  # 紀錄空孔最終位置
        idz_record = {}
        num_repetitions = 0
        while num_repetitions <= self.repetitions:
            # 進行重複取樣
            walkers_record[num_repetitions] = {}
            lattice_record[num_repetitions] = {}
            v_position_record[num_repetitions] = {}
            idz_recording = np.empty((self.numofvacancy + 1, 2), dtype=int)
            atom_list_left, atom_list_right = self.homo_fccatomlist()
            sc_lattice = self.homo_fccdiffusioncouple(atom_list_left, atom_list_right)
            left_idz_layer, right_idz_layer = (
                self.couple_length - 1,
                self.couple_length + 1,
            )
            success = 0
            while success <= self.numofvacancy:
                # 再擺放前，先計算flux profile，再根據profile，找出idz range
                flux_result = self.calc_concentration(sc_lattice)
                left_idz_layer, right_idz_layer = self.find_gradient_layer(
                    left_idz_layer, right_idz_layer, flux_result, 1
                )
                idz_recording[success][0], idz_recording[success][1] = (
                    left_idz_layer,
                    right_idz_layer,
                )
                idz_idnex = self.idzboundary_index(left_idz_layer, right_idz_layer)
                print(
                    f"Now in {self.alloy} {num_repetitions} {success} with {left_idz_layer, right_idz_layer}"
                )
                # Lattice construct and randomly place atom onto it.
                lattice, v_position_index, remove_atom_type = self.vacancy_create(
                    sc_lattice
                )
                vacancy_record = np.empty((self.steps, 3), dtype=float)
                position_recording = np.empty((self.steps,), dtype=int)  # 紀錄空孔位置
                try:
                    for step in range(self.steps):
                        if self.check_in_gradient_layer(v_position_index, idz_idnex):
                            neighbor_index, neighbor = self.find_neighbor(
                                lattice, v_position_index
                            )
                            # Determine the exchange diretion
                            exchange_probability = self.prob(neighbor)
                            (
                                v_after_index,
                                v_movement_vector,
                            ) = self.determine_exchange(
                                neighbor_index, exchange_probability
                            )
                            # Perform the vacancy exchangement with chosen atom and direciton
                            lattice, v_position_index = self.perform_exchnage(
                                lattice, v_position_index, v_after_index
                            )

                            # Record the exchange moment
                            vacancy_record = self.record_exchange(
                                vacancy_record, step, v_movement_vector
                            )
                            position_recording[step] = v_position_index
                        else:
                            vacancy_record = vacancy_record[0:step]
                            position_recording = position_recording[0:step]
                            break

                except Exception as e:
                    print(e)
                    if success in lattice_record[num_repetitions]:
                        del lattice_record[num_repetitions][success]
                    continue

                vacancy_trace = np.cumsum(vacancy_record, axis=0)
                walkers_record[num_repetitions]["x" + str(success)] = vacancy_trace[
                    :, 0
                ]
                walkers_record[num_repetitions]["y" + str(success)] = vacancy_trace[
                    :, 1
                ]
                walkers_record[num_repetitions]["z" + str(success)] = vacancy_trace[
                    :, 2
                ]
                walkers_record[num_repetitions]["SD" + str(success)] = np.sum(
                    np.square(vacancy_trace), axis=1
                )
                v_position_record[num_repetitions][success] = position_recording

                sc_lattice = self.rep_lattice_record(
                    lattice, v_position_index, remove_atom_type
                )
                if success % 10 == 0:
                    lattice_record[num_repetitions][success] = np.copy(sc_lattice)
                success += 1
            idz_record[num_repetitions] = idz_recording.reshape(-1)
            num_repetitions += 1
        vacancy_trace = pd.DataFrame(walkers_record)
        filtered_lattice_record = pd.DataFrame(lattice_record)
        v_position = pd.DataFrame(v_position_record)
        idz = pd.DataFrame(idz_record)

        return vacancy_trace, filtered_lattice_record, v_position, idz

    def save_data(
        self, parameter, vacancy_record, lattice_record, v_position_record, idz
    ):
        """
        Save the data with DataFrame in form of json
        """
        currentdir = os.path.abspath(os.curdir)
        dirname = f"fcc_serial_dynamicIDZ_diffpro_data_size{parameter[-1]}_numofvacancy{parameter[-3]}_{parameter[1]}"
        # file_dir = r"/work1/u3284480/"
        dirpath = os.path.join(currentdir, "serial", "0.02", f"{self.crystal}", dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        os.chdir(dirpath)
        data_path = os.path.join(dirpath, f"{self.alloy}")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        os.chdir(data_path)
        lattice_record.to_json(
            f"Lattice record of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        vacancy_record.to_json(
            f"Vacancy trace of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        v_position_record.to_json(
            f"Vacancy position of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        idz.to_json(
            f"idz of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        with open(f"{self.alloy}_Parameter.txt", "w") as fh:
            parameter_name = [
                "alloy",
                "fluc",
                "elements",
                "concentration",
                "exchangerate",
                "ratio",
                "steps",
                "numofvacancy",
                "repetitions",
                "lattice_size",
            ]
            for name, para in zip(parameter_name, parameter):
                fh.write(f"{name}:{para}\n")
        os.chdir(currentdir)

        return


class BCCInterdiffusion:
    def __init__(
        self,
        crystal,
        alloy,
        elements,
        concentration,
        exchangerate,
        aspectratio=[4, 1],
        steps=100000,
        numofvacancy=100,
        repetitions=100,
        lattice_size=1000,
    ):
        self.steps = int(steps)
        self.crystal = crystal
        self.alloy = alloy
        self.numofvacancy = int(numofvacancy)
        self.exchangerate = exchangerate
        self.const = 0.08625  # kT
        self.repetitions = repetitions
        self.elements = elements
        self.lattice_size = int(lattice_size)
        self.plane_length = int(aspectratio[1] * self.lattice_size)
        self.couple_length = int(aspectratio[0] * self.lattice_size)
        self.plane_atom_num = self.plane_length**2
        self.directions = np.array(
            [
                [-0.5, -0.5, 0.5],  # upeshell up
                [-0.5, -0.5, -0.5],  # upeshell down
                [-0.5, 0.5, 0.5],  # right up
                [-0.5, 0.5, -0.5],  # right down
                [0.5, -0.5, 0.5],  # left up
                [0.5, -0.5, -0.5],  # left up
                [0.5, 0.5, 0.5],  # downshell up
                [0.5, 0.5, -0.5],  # downshell down
            ]
        )
        self.concentration = concentration
        (
            self.upboundary,
            self.downboundary,
            self.frontboundary,
            self.backboundary,
            self.downfrontboundary,
            self.upfrontboundary,
            self.downbackboundary,
            self.upbackboundary,
            self.edgeboundary,
            self.cornerboundary,
            self.leftrightboundary,
        ) = self.boundary_make()

    def homo_bccatomlist(self):
        # Left of diffusion couple
        planar_num_of_atoms_left = [
            round(self.plane_atom_num * element_concentration)
            for element_concentration in self.concentration[0]
        ]
        planar_num_of_atoms_left[-1] = self.plane_atom_num - sum(
            planar_num_of_atoms_left[0:-1]
        )
        planar_atom_list_left = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(
                    self.elements[0], planar_num_of_atoms_left
                )
            ]
        )
        planar_num_of_atoms_right = [
            round(self.plane_atom_num * element_concentration)
            for element_concentration in self.concentration[1]
        ]
        planar_num_of_atoms_right[-1] = self.plane_atom_num - sum(
            planar_num_of_atoms_right[0:-1]
        )
        planar_atom_list_right = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(
                    self.elements[1], planar_num_of_atoms_right
                )
            ]
        )
        return planar_atom_list_left, planar_atom_list_right

    def homo_bccdiffusioncouple(self, planar_atom_list_left, planar_atom_list_right):
        # Left diffusion coouple
        left_couple = np.array([])
        for _ in range(self.couple_length):
            planer_atom = planar_atom_list_left
            np.random.shuffle(planer_atom)
            left_couple = np.concatenate(([left_couple, planer_atom]))

        right_couple = np.array([])
        for _ in range(self.couple_length):
            planer_atom = planar_atom_list_right
            np.random.shuffle(planer_atom)
            right_couple = np.concatenate(([right_couple, planer_atom]))
        lattice = np.concatenate([left_couple, right_couple])
        return lattice.astype(int)

    def boundary_make(self):
        couple_total_index = 2 * self.couple_length * (self.plane_atom_num)

        left_boundary = np.arange(self.plane_atom_num)
        back_boundary = np.arange(
            2 * self.plane_length - 1, couple_total_index, 2 * self.plane_length
        )
        down_boundary = np.concatenate(
            [
                np.arange(i, i + self.plane_length, self.plane_length)
                for i in range(0, couple_total_index, self.plane_atom_num)
            ]
        )
        up_boundary = np.concatenate(
            [
                np.arange(i, i + self.plane_length)
                for i in range(
                    self.plane_atom_num - self.plane_length,
                    couple_total_index,
                    self.plane_atom_num,
                )
            ]
        )
        front_boundary = np.arange(0, couple_total_index, 2 * self.plane_length)
        right_boundary = np.arange(
            couple_total_index - self.plane_atom_num, couple_total_index
        )

        # Edge Boundary
        down_front_boundary = np.arange(0, couple_total_index, self.plane_atom_num)
        down_back_boundary = np.arange(
            self.plane_length - 1, couple_total_index, self.plane_atom_num
        )
        up_front_boundary = np.arange(
            self.plane_atom_num - self.plane_length,
            couple_total_index,
            self.plane_atom_num,
        )
        up_back_boundary = np.arange(
            self.plane_atom_num - 1, couple_total_index, self.plane_atom_num
        )
        corner_boundary = np.unique(
            np.concatenate(
                [
                    down_front_boundary,
                    down_back_boundary,
                    up_front_boundary,
                    up_back_boundary,
                ]
            )
        )
        edge_boundary = np.unique(
            np.concatenate(
                [
                    up_boundary,
                    down_boundary,
                    front_boundary,
                    back_boundary,
                    corner_boundary,
                ]
            )
        )
        left_right_boundary = np.unique(np.concatenate([left_boundary, right_boundary]))

        return (
            up_boundary,
            down_boundary,
            front_boundary,
            back_boundary,
            down_front_boundary,
            up_front_boundary,
            down_back_boundary,
            up_back_boundary,
            edge_boundary,
            corner_boundary,
            left_right_boundary,
        )

    def check_bound(self, position_index):
        return np.any(np.isin(position_index, self.leftrightboundary))

    def calc_concentration(self, lattice):
        composition_record = {}
        for i, x in enumerate(
            range(0, 2 * self.couple_length * self.plane_atom_num, self.plane_atom_num)
        ):
            atoms = lattice[x : x + 1 * self.plane_atom_num]
            element, concentration = np.unique(atoms, return_counts=True)
            composition_record[i] = dict(zip(element, concentration / len(atoms)))
        concentration_df = pd.DataFrame.from_dict(composition_record, orient="index")
        concentration_df = concentration_df.fillna(0)
        gradient_result = pd.DataFrame()
        # First derivative of concentration
        gradient_result[f"Flux"] = (
            -concentration_df.iloc[:, 0].diff(periods=-1)
            + concentration_df.iloc[:, 0].diff()
        ) * 0.5
        gradient_result.fillna(0)
        return gradient_result

    def find_gradient_layer(
        self, left_idz_layer, right_idz_layer, concentration_profile, bear_layer
    ):
        left_before_con = concentration_profile.iloc[left_idz_layer, 0]
        left_after_con = concentration_profile.iloc[left_idz_layer - bear_layer, 0]
        right_before_con = concentration_profile.iloc[right_idz_layer, 0]
        right_after_con = concentration_profile.iloc[right_idz_layer + bear_layer, 0]

        if left_after_con - left_before_con >= 0.02:
            left_idz_layer -= 1
        if right_before_con - right_after_con >= 0.02:
            right_idz_layer += 1

        return left_idz_layer, right_idz_layer

    def idzboundary_index(self, left_idz_layer, right_idz_layer):
        return np.arange(
            self.plane_atom_num * self.couple_length
            + ((left_idz_layer - self.couple_length) - 1) * self.plane_atom_num,
            self.plane_atom_num * self.couple_length
            + (right_idz_layer - self.couple_length) * self.plane_atom_num,
        )

    def vacancy_create(self, lattice):
        """
        Create a vacancy in center of lattice site, and return both new lattice and index of vancancy position
        """
        center = int(
            self.couple_length * self.plane_atom_num - (self.plane_atom_num // 2)
        )
        remove_atom_type = lattice[center]
        lattice[center] = 0
        return lattice, center, remove_atom_type

    def check_in_gradient_layer(self, position_index, idzboundary):
        return np.any(np.isin(position_index, idzboundary))

    def v_position_layer(self, v_position):
        return v_position // self.plane_atom_num

    def find_neighbor_adjust(self, v_position):
        layerlocation = self.v_position_layer(v_position)
        previsous_atom_num = (layerlocation) * self.plane_atom_num
        row = (v_position - previsous_atom_num) // self.plane_length
        if np.any(np.isin(v_position, self.edgeboundary)):
            if np.any(np.isin(v_position, self.cornerboundary)):
                raise AssertionError("Reach corner boundary")
            elif np.any(np.isin(v_position, self.upboundary)):
                if layerlocation % 2:  # odd layer
                    adjust = [
                        [-2, 1, 0],
                        [-2, 1, 1],
                        [-1, 1, 0],
                        [-1, 1, 1],
                        [0, -1, 0],
                        [0, -1, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                    ]
                else:
                    adjust = [
                        [-1, -1, 0],
                        [-1, -1, 1],
                        [-1, 1, 0],
                        [-1, 1, 1],
                        [0, -1, 0],
                        [0, -1, 1],
                        [1, -1, 0],
                        [1, -1, 1],
                    ]
            elif np.any(np.isin(v_position, self.downboundary)):
                if layerlocation % 2:  # odd layer
                    adjust = [
                        [-1, 1, -1],
                        [-1, 1, 0],
                        [0, 1, -1],
                        [0, 1, 0],
                        [1, -1, -1],
                        [1, -1, 0],
                        [1, 1, -1],
                        [1, 1, 0],
                    ]
                else:
                    adjust = [
                        [0, -1, -1],
                        [0, -1, 0],
                        [0, 1, -1],
                        [0, 1, 0],
                        [1, -1, -1],
                        [1, -1, 0],
                        [2, -1, -1],
                        [2, -1, 0],
                    ]
            elif np.any(np.isin(v_position, self.frontboundary)):
                if layerlocation % 2:  # odd layer
                    adjust = [
                        [-1, 2, -1],
                        [-1, 1, 0],
                        [0, 2, -1],
                        [0, 1, 0],
                        [0, -1, -1],
                        [0, -1, 0],
                        [1, 2, -1],
                        [1, 1, 0],
                    ]
                else:
                    adjust = [
                        [-1, 0, -1],
                        [-1, -1, 0],
                        [0, 2, -1],
                        [0, 1, 0],
                        [0, 0, -1],
                        [0, -1, 0],
                        [1, -1, 0],
                        [1, 0, -1],
                    ]
            elif np.any(np.isin(v_position, self.backboundary)):
                if layerlocation % 2:  # odd layer
                    adjust = [
                        [-1, 1, 0],
                        [-1, 0, 1],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, -1, 0],
                        [0, -2, 1],
                        [1, 1, 0],
                        [1, 0, 1],
                    ]
                else:
                    adjust = [
                        [-1, -1, 0],
                        [-1, -2, 1],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, -1, 0],
                        [0, -2, 1],
                        [1, -1, 0],
                        [1, -2, 1],
                    ]
        elif layerlocation % 2:  # odd layer
            if row % 2:
                adjust = [
                    [-1, 1, 0],
                    [-1, 1, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, -1, 0],
                    [0, -1, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ]
            else:
                adjust = [
                    [-1, 1, -1],
                    [-1, 1, 0],
                    [0, 1, -1],
                    [0, 1, 0],
                    [0, -1, -1],
                    [0, -1, 0],
                    [1, 1, -1],
                    [1, 1, 0],
                ]
        else:  # even layer
            if row % 2:
                adjust = [
                    [-1, -1, 0],
                    [-1, -1, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, -1, 0],
                    [0, -1, 1],
                    [1, -1, 0],
                    [1, -1, 1],
                ]
            else:
                adjust = [
                    [-1, -1, -1],
                    [-1, -1, 0],
                    [0, 1, -1],
                    [0, 1, 0],
                    [0, -1, -1],
                    [0, -1, 0],
                    [1, -1, -1],
                    [1, -1, 0],
                ]

        return adjust

    def find_neighbor(self, lattice, v_position_index):
        adjust = self.find_neighbor_adjust(v_position_index)
        neighbor_index = []
        for a in adjust:
            neighbor_index.append(
                v_position_index
                + self.plane_atom_num * a[0]
                + self.plane_length * a[1]
                + a[2]
            )
        neighbor = np.array([lattice[indexing] for indexing in neighbor_index])
        return neighbor_index, neighbor

    def prob(self, neighbor_atom):
        """
        Calculate exchange probability for vacancy with 1-st shell atom, which
        is base on migration barrier randomly drawn from distribution.
        """
        ex_probability = np.array(
            [self.exchangerate[int(atom) - 1] for atom in neighbor_atom]
        )
        nor_ex_probability = ex_probability / sum(ex_probability)

        return nor_ex_probability

    def determine_exchange(self, neighbor_index, exchange_prob):
        """
        Determine the exchange direction
        """
        chosen_direction = np.random.choice(self.directions.shape[0], p=exchange_prob)
        chosen_index = neighbor_index[chosen_direction]
        chosen_vector = self.directions[chosen_direction]

        return chosen_index, chosen_vector

    def perform_exchnage(self, lattice, v_position, change_direction_index):
        """
        Perform exchange
        """
        lattice[v_position], lattice[change_direction_index] = (
            lattice[change_direction_index],
            lattice[v_position],
        )
        v_position = change_direction_index
        return lattice, v_position

    def record_exchange(self, record, step, direction):
        record[step] = direction[0:3]
        return record

    def rep_lattice_record(self, lattice, v_position, remove_atom_type):
        lattice[v_position] = remove_atom_type
        return lattice

    def random_walk(self):
        walkers_record = {}  # 紀錄vacancy移動軌跡
        lattice_record = {}  # 紀錄lattice
        v_position_record = {}  # 紀錄空孔最終位置
        idz_record = {}
        num_repetitions = 0
        while num_repetitions <= self.repetitions:
            # 進行重複取樣
            walkers_record[num_repetitions] = {}
            lattice_record[num_repetitions] = {}
            v_position_record[num_repetitions] = {}
            idz_recording = np.empty((self.numofvacancy + 1, 2), dtype=int)
            atom_list_left, atom_list_right = self.homo_bccatomlist()
            sc_lattice = self.homo_bccdiffusioncouple(atom_list_left, atom_list_right)
            left_idz_layer, right_idz_layer = (
                self.couple_length - 1,
                self.couple_length + 1,
            )
            success = 0
            while success <= self.numofvacancy:
                # 再擺放前，先計算flux profile，再根據profile，找出idz range
                flux_result = self.calc_concentration(sc_lattice)
                left_idz_layer, right_idz_layer = self.find_gradient_layer(
                    left_idz_layer, right_idz_layer, flux_result, 1
                )
                idz_recording[success][0], idz_recording[success][1] = (
                    left_idz_layer,
                    right_idz_layer,
                )
                idz_idnex = self.idzboundary_index(left_idz_layer, right_idz_layer)
                print(
                    f"Now in {self.alloy} {num_repetitions} {success} with {left_idz_layer, right_idz_layer}"
                )
                # Lattice construct and randomly place atom onto it.
                lattice, v_position_index, remove_atom_type = self.vacancy_create(
                    sc_lattice
                )
                vacancy_record = np.empty((self.steps, 3), dtype=float)
                position_recording = np.empty((self.steps,), dtype=int)  # 紀錄空孔位置
                try:
                    for step in range(self.steps):
                        if self.check_in_gradient_layer(v_position_index, idz_idnex):
                            neighbor_index, neighbor = self.find_neighbor(
                                lattice, v_position_index
                            )
                            # Determine the exchange diretion
                            exchange_probability = self.prob(neighbor)
                            (
                                v_after_index,
                                v_movement_vector,
                            ) = self.determine_exchange(
                                neighbor_index, exchange_probability
                            )
                            # Perform the vacancy exchangement with chosen atom and direciton
                            lattice, v_position_index = self.perform_exchnage(
                                lattice, v_position_index, v_after_index
                            )

                            # Record the exchange moment
                            vacancy_record = self.record_exchange(
                                vacancy_record, step, v_movement_vector
                            )
                            position_recording[step] = v_position_index
                        else:
                            vacancy_record = vacancy_record[0:step]
                            position_recording = position_recording[0:step]
                            break

                except Exception as e:
                    print(e)
                    if success in lattice_record[num_repetitions]:
                        del lattice_record[num_repetitions][success]
                    continue

                vacancy_trace = np.cumsum(vacancy_record, axis=0)
                walkers_record[num_repetitions]["x" + str(success)] = vacancy_trace[
                    :, 0
                ]
                walkers_record[num_repetitions]["y" + str(success)] = vacancy_trace[
                    :, 1
                ]
                walkers_record[num_repetitions]["z" + str(success)] = vacancy_trace[
                    :, 2
                ]
                walkers_record[num_repetitions]["SD" + str(success)] = np.sum(
                    np.square(vacancy_trace), axis=1
                )
                v_position_record[num_repetitions][success] = position_recording

                sc_lattice = self.rep_lattice_record(
                    lattice, v_position_index, remove_atom_type
                )
                if success % 10 == 0:
                    lattice_record[num_repetitions][success] = np.copy(sc_lattice)
                success += 1
            idz_record[num_repetitions] = idz_recording.reshape(-1)
            num_repetitions += 1
        vacancy_trace = pd.DataFrame(walkers_record)
        filtered_lattice_record = pd.DataFrame(lattice_record)
        v_position = pd.DataFrame(v_position_record)
        idz = pd.DataFrame(idz_record)

        return vacancy_trace, filtered_lattice_record, v_position, idz

    def save_data(
        self, parameter, vacancy_record, lattice_record, v_position_record, idz
    ):
        """
        Save the data with DataFrame in form of json
        """
        currentdir = os.path.abspath(os.curdir)
        dirname = f"bcc_serial_dynamicIDZ_diffpro_data_size{parameter[-1]}_numofvacancy{parameter[-3]}_{parameter[1]}"
        # file_dir = r"/work1/u3284480/"
        dirpath = os.path.join(currentdir, "serial", "0.02", f"{self.crystal}", dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        os.chdir(dirpath)
        data_path = os.path.join(dirpath, f"{self.alloy}")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        os.chdir(data_path)
        lattice_record.to_json(
            f"Lattice record of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        vacancy_record.to_json(
            f"Vacancy trace of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        v_position_record.to_json(
            f"Vacancy position of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        idz.to_json(
            f"idz of {self.alloy} with {self.numofvacancy} json.gz",
            compression="gzip",
        )
        with open(f"{self.alloy}_Parameter.txt", "w") as fh:
            parameter_name = [
                "alloy",
                "fluc",
                "elements",
                "concentration",
                "exchangerate",
                "ratio",
                "steps",
                "numofvacancy",
                "repetitions",
                "lattice_size",
            ]
            for name, para in zip(parameter_name, parameter):
                fh.write(f"{name}:{para}\n")
        os.chdir(currentdir)

        return


def task(alloy_para):
    (
        alloy,
        fluc,
        elements,
        concentration,
        exchange_probability,
        ratio,
        steps,
        repetitions,
        trail,
        lattice_size,
    ) = alloy_para
    scwalker = SCInterdiffusion(
        "scLattice",
        alloy,
        elements,
        concentration,
        exchange_probability,
        ratio,
        steps,
        repetitions,
        trail,
        lattice_size,
    )
    vacancy_trace, walker_record, position_record, error = scwalker.random_walk()
    scwalker.save_data(alloy_para, vacancy_trace, walker_record, position_record, error)
    print(f"SC is done")
    (
        alloy,
        fluc,
        elements,
        concentration,
        exchange_probability,
        ratio,
        steps,
        repetitions,
        trail,
        lattice_size,
    ) = alloy_para
    fccwalker = FCCInterdiffusion(
        "fccLattice",
        alloy,
        elements,
        concentration,
        exchange_probability,
        ratio,
        steps,
        repetitions,
        trail,
        lattice_size,
    )
    vacancy_trace, walker_record, position_record, error = fccwalker.random_walk()
    fccwalker.save_data(
        alloy_para, vacancy_trace, walker_record, position_record, error
    )
    print(f"FCC is done")
    (
        alloy,
        fluc,
        elements,
        concentration,
        exchange_probability,
        ratio,
        steps,
        repetitions,
        trail,
        lattice_size,
    ) = alloy_para
    bccwalker = BCCInterdiffusion(
        "bccLattice",
        alloy,
        elements,
        concentration,
        exchange_probability,
        ratio,
        steps,
        repetitions,
        trail,
        lattice_size,
    )
    vacancy_trace, walker_record, position_record, error = bccwalker.random_walk()
    bccwalker.save_data(
        alloy_para, vacancy_trace, walker_record, position_record, error
    )


def parameter(
    nb_min,
    nb_max,
    wb_min,
    wb_max,
    ratio,
    steps=100000,
    numofvacancy=100,
    rep=100,
    lattice_size=1000,
):

    nb_couple_MtoM_5dev = {
        "A1A2": {
            "fluc": f"nb{nb_max}_5fluc",
            "elements": [[1, 2], [1, 2]],
            "concentration": [[1, 0], [0, 1]],
            "exchangeprobability": np.linspace(nb_max, nb_min, 2),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "AB1AB2": {
            "fluc": f"nb{nb_max}_5fluc",
            "elements": [[1, 2], [1, 2]],
            "concentration": [[0.625, 0.375], [0.375, 0.625]],
            "exchangeprobability": np.linspace(nb_max, nb_min, 2),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABC1ABC2": {
            "fluc": f"nb{nb_max}_5fluc",
            "elements": [[1, 2, 3], [1, 2, 3]],
            "concentration": [[0.4125, 0.34, 0.2475], [0.2475, 0.34, 0.4125]],
            "exchangeprobability": np.linspace(nb_max, nb_min, 3),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABCD1ABCD2": {
            "fluc": f"nb{nb_max}_5fluc",
            "elements": [[1, 2, 3, 4], [1, 2, 3, 4]],
            "concentration": [
                [0.3125, 0.25, 0.25, 0.1875],
                [0.1875, 0.25, 0.25, 0.3125],
            ],
            "exchangeprobability": np.linspace(nb_max, nb_min, 4),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABCDE1ABCDE2": {
            "fluc": f"nb{nb_max}_5fluc",
            "elements": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            "concentration": [
                [0.25, 0.2, 0.2, 0.2, 0.15],
                [0.15, 0.2, 0.2, 0.2, 0.25],
            ],
            "exchangeprobability": np.linspace(nb_max, nb_min, 5),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
    }
    nb_couple_MtoM_10dev = {
        "A1A2": {
            "fluc": f"nb{nb_max}_10fluc",
            "elements": [[1, 2], [1, 2]],
            "concentration": [[1, 0], [0, 1]],
            "exchangeprobability": np.linspace(nb_max, nb_min, 2),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "AB1AB2": {
            "fluc": f"nb{nb_max}_10fluc",
            "elements": [[1, 2], [1, 2]],
            "concentration": [[0.75, 0.25], [0.25, 0.75]],
            "exchangeprobability": np.linspace(nb_max, nb_min, 2),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABC1ABC2": {
            "fluc": f"nb{nb_max}_10fluc",
            "elements": [[1, 2, 3], [1, 2, 3]],
            "concentration": [[0.495, 0.34, 0.165], [0.165, 0.34, 0.495]],
            "exchangeprobability": np.linspace(nb_max, nb_min, 3),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABCD1ABCD2": {
            "fluc": f"nb{nb_max}_10fluc",
            "elements": [[1, 2, 3, 4], [1, 2, 3, 4]],
            "concentration": [
                [0.375, 0.25, 0.25, 0.125],
                [0.125, 0.25, 0.25, 0.375],
            ],
            "exchangeprobability": np.linspace(nb_max, nb_min, 4),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABCDE1ABCDE2": {
            "fluc": f"nb{nb_max}_10fluc",
            "elements": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            "concentration": [
                [0.3, 0.2, 0.2, 0.2, 0.1],
                [0.1, 0.2, 0.2, 0.2, 0.1],
            ],
            "exchangeprobability": np.linspace(nb_max, nb_min, 5),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
    }
    wb_couple_MtoM_5dev = {
        "A1A2": {
            "fluc": f"wb{wb_max}_5fluc",
            "elements": [[1, 2], [1, 2]],
            "concentration": [[1, 0], [0, 1]],
            "exchangeprobability": np.linspace(wb_max, wb_min, 2),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "AB1AB2": {
            "fluc": f"wb{wb_max}_5fluc",
            "elements": [[1, 2], [1, 2]],
            "concentration": [[0.625, 0.375], [0.375, 0.625]],
            "exchangeprobability": np.linspace(wb_max, wb_min, 2),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABC1ABC2": {
            "fluc": f"wb{wb_max}_5fluc",
            "elements": [[1, 2, 3], [1, 2, 3]],
            "concentration": [[0.4125, 0.34, 0.2475], [0.2475, 0.34, 0.4125]],
            "exchangeprobability": np.linspace(wb_max, wb_min, 3),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABCD1ABCD2": {
            "fluc": f"wb{wb_max}_5fluc",
            "elements": [[1, 2, 3, 4], [1, 2, 3, 4]],
            "concentration": [
                [0.3125, 0.25, 0.25, 0.1875],
                [0.1875, 0.25, 0.25, 0.3125],
            ],
            "exchangeprobability": np.linspace(wb_max, wb_min, 4),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABCDE1ABCDE2": {
            "fluc": f"wb{wb_max}_5fluc",
            "elements": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            "concentration": [
                [0.25, 0.2, 0.2, 0.2, 0.15],
                [0.15, 0.2, 0.2, 0.2, 0.25],
            ],
            "exchangeprobability": np.linspace(wb_max, wb_min, 5),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
    }
    wb_couple_MtoM_10dev = {
        "A1A2": {
            "fluc": f"wb{wb_max}_10fluc",
            "elements": [[1, 2], [1, 2]],
            "concentration": [[1, 0], [0, 1]],
            "exchangeprobability": np.linspace(wb_max, wb_min, 2),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "AB1AB2": {
            "fluc": f"wb{wb_max}_10fluc",
            "elements": [[1, 2], [1, 2]],
            "concentration": [[0.75, 0.25], [0.25, 0.75]],
            "exchangeprobability": np.linspace(wb_max, wb_min, 2),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABC1ABC2": {
            "fluc": f"wb{wb_max}_10fluc",
            "elements": [[1, 2, 3], [1, 2, 3]],
            "concentration": [[0.495, 0.34, 0.165], [0.165, 0.34, 0.495]],
            "exchangeprobability": np.linspace(wb_max, wb_min, 3),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABCD1ABCD2": {
            "fluc": f"wb{wb_max}_10fluc",
            "elements": [[1, 2, 3, 4], [1, 2, 3, 4]],
            "concentration": [
                [0.375, 0.25, 0.25, 0.125],
                [0.125, 0.25, 0.25, 0.375],
            ],
            "exchangeprobability": np.linspace(wb_max, wb_min, 4),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
        "ABCDE1ABCDE2": {
            "fluc": f"wb{wb_max}_10fluc",
            "elements": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            "concentration": [
                [0.3, 0.2, 0.2, 0.2, 0.1],
                [0.1, 0.2, 0.2, 0.2, 0.3],
            ],
            "exchangeprobability": np.linspace(wb_max, wb_min, 5),
            "ratio": ratio,
            "steps": steps,
            "numofvacancy": numofvacancy,
            "rep": rep,
            "lattice": lattice_size,
        },
    }
    return (
        nb_couple_MtoM_5dev,
        nb_couple_MtoM_10dev,
        wb_couple_MtoM_5dev,
        wb_couple_MtoM_10dev,
    )


if __name__ == "__main__":
    pool = mp.Pool(processes=10)
    nbmin, nbmax, wbmin, wbmax, ratio, steps, numofvacancy, repetition, lattice_size = (
        1,
        2.5,
        1,
        5,
        [10, 1],
        2000,
        100,
        100,
        10,
    )
    alloyparater = parameter(
        nbmin, nbmax, wbmin, wbmax, ratio, steps, numofvacancy, repetition, lattice_size
    )
    value = []
    for diffusioncouple in alloyparater:
        for alloysys in diffusioncouple:
            alloysimualtionparater = []
            alloysimualtionparater.append(alloysys)
            for alloysyspara in diffusioncouple[alloysys].items():
                alloysimualtionparater.append(alloysyspara[1])
            value.append(alloysimualtionparater)
    print(value)
    res = pool.map(task, value)
