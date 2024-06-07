import os, time, json, gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import special


class SCInterdiffusionGradient:
    def __init__(
        self,
        crystalstructure,
        fluc,
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
        self.alloy = alloy
        self.fluc = fluc
        self.crystalstructure = crystalstructure
        self.numofvacancy = int(numofvacancy)
        self.repetitions = repetitions
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

    def erf_concentration_profile(self, layer_num):
        adjust_table = {
            "5fluc": {
                "A1A2": 0.5,
                "AB1AB2": 2,
                "ABC1ABC2": 3,
                "ABCD1ABCD2": 4,
                "ABCDE1ABCDE2": 5,
            },
            "10fluc": {
                "A1A2": 1,
                "AB1AB2": 2,
                "ABC1ABC2": 3,
                "ABCD1ABCD2": 4,
                "ABCDE1ABCDE2": 5,
            },
        }
        adjust = adjust_table[self.fluc.split("_")[1]][self.alloy]
        # 產生A
        left_a = self.concentration[0][0]
        right_a = self.concentration[1][0]
        position = np.linspace(-adjust / 2, adjust / 2, layer_num)
        conc_gradient_a = 0.5 * (left_a + right_a) + 0.5 * (
            right_a - left_a
        ) * special.erf(position)
        # 產生B
        left_b = self.concentration[0][-1]
        right_b = self.concentration[1][-1]
        conc_gradient_b = 0.5 * (left_b + right_b) + 0.5 * (
            right_b - left_b
        ) * special.erf(position)

        return list(zip(conc_gradient_a, conc_gradient_b))

    def scatomlist(self):
        total_alloy_atoms = self.couple_length * self.plane_atom_num

        # left end of diffusion couple
        num_of_atoms_left = [
            round(total_alloy_atoms * element_concentration)
            for element_concentration in self.concentration[0]
        ]
        num_of_atoms_left[-1] = total_alloy_atoms - sum(num_of_atoms_left[0:-1])
        atom_list_left = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(self.elements[0], num_of_atoms_left)
            ]
        )

        # right end of diffusion couple
        num_of_atoms_right = [
            round(total_alloy_atoms * element_concentration)
            for element_concentration in self.concentration[1]
        ]
        num_of_atoms_right[-1] = total_alloy_atoms - sum(num_of_atoms_right[0:-1])
        atom_list_right = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(self.elements[1], num_of_atoms_right)
            ]
        )
        return atom_list_left, atom_list_right

    def scdiffusioncouple(self, atom_list_left, atom_list_right, conc_gradient):
        # 重新排列原子
        copy_atom_list_left = np.copy(atom_list_left)
        copy_atom_list_right = np.copy(atom_list_right)
        np.random.shuffle(copy_atom_list_left)
        np.random.shuffle(copy_atom_list_right)

        diffusion_couple_array = np.concatenate(
            [copy_atom_list_left, copy_atom_list_right]
        )

        # 根據設定erf 調整
        half_layer_num = len(conc_gradient) // 2
        for i, layer in enumerate(range(-half_layer_num, half_layer_num)):

            modified_conc = self.concentration[0]
            modified_conc[0], modified_conc[-1] = (
                conc_gradient[i][0],
                conc_gradient[i][-1],
            )
            num_of_atoms = [
                round(self.plane_atom_num * element_concentration)
                for element_concentration in modified_conc
            ]
            num_of_atoms[-1] = self.plane_atom_num - sum(num_of_atoms[0:-1])
            modified_atom_list = np.concatenate(
                [
                    np.full(element_nums, element, dtype=int)
                    for element, element_nums in zip(self.elements[0], num_of_atoms)
                ]
            )
            np.random.shuffle(modified_atom_list)
            diffusion_couple_array[
                ((self.couple_length + layer) * self.plane_atom_num) : (
                    (self.couple_length + layer + 1) * self.plane_atom_num
                )
            ] = modified_atom_list

        return diffusion_couple_array

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

    def vacancy_create(self, lattice_array):
        """
        Create a vacancy in center of lattice site, and return both new lattice and index of vancancy position
        """
        lattice = np.copy(lattice_array)
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
        num_repetitions = 0
        atom_list_left, atom_list_right = self.scatomlist()
        erfc = self.erf_concentration_profile(self.couple_length * 2)
        edge_fail = 0
        boundary_fail = 0
        while num_repetitions <= self.repetitions:
            # 進行重複取樣
            walkers_record[num_repetitions] = {}
            v_position_record[num_repetitions] = {}
            bcc_lattice = self.scdiffusioncouple(atom_list_left, atom_list_right, erfc)
            lattice_record[num_repetitions] = bcc_lattice
            success = 0
            while success <= self.numofvacancy:
                # 持續擺放vacancy
                print(f"Now in {num_repetitions} {success}")
                # Lattice construct and randomly place atom onto it.
                lattice, v_position_index, remove_atom_type = self.vacancy_create(
                    bcc_lattice
                )
                vacancy_record = np.empty((self.steps, 3), dtype=float)
                position_recording = []  # 紀錄空孔位置
                check_in_layer = np.empty((self.steps,), dtype=int)
                try:
                    for step in range(self.steps):
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

                        if (step + 1) % 10 == 0:
                            position_recording.append(v_position_index)
                        check_in_layer[step] = v_position_index
                    if self.check_bound(check_in_layer):
                        print("out of layer")
                        continue
                except IndexError as e1:
                    print(e1)
                    boundary_fail += 1
                    continue
                except AssertionError as e2:
                    print(e2)
                    edge_fail += 1
                    continue

                if not self.check_bound(check_in_layer):
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

                    success += 1
            num_repetitions += 1
        vacancy_trace = pd.DataFrame(walkers_record)
        filtered_lattice_record = pd.DataFrame(lattice_record)
        v_position = pd.DataFrame(v_position_record)
        error = (edge_fail, boundary_fail)
        return vacancy_trace, filtered_lattice_record, v_position, error

    def save_data(
        self, parameter, vacancy_record, lattice_record, v_position_record, error
    ):
        """
        Save the data with DataFrame in form of json
        """
        currentdir = os.path.abspath(os.curdir)
        dirname = f"sc_gradient_diffpro_data_size{parameter[-1]}_numofvacancy{parameter[-3]}_{parameter[1]}"
        # file_dir = r"/work1/u3284480/"
        dirpath = os.path.join(
            currentdir, "gradient", f"{self.crystalstructure}", dirname
        )
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
            fh.write(f"error:{error}")
        os.chdir(currentdir)

        return


class FCCInterdiffusionGradient:
    def __init__(
        self,
        crystalstructure,
        fluc,
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
        self.crystalstructure = crystalstructure
        self.fluc = fluc
        self.alloy = alloy
        self.numofvacancy = int(numofvacancy)
        self.repetitions = repetitions
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

    def erf_concentration_profile(self, layer_num):
        adjust_table = {
            "5fluc": {
                "A1A2": 0.5,
                "AB1AB2": 2,
                "ABC1ABC2": 3,
                "ABCD1ABCD2": 4,
                "ABCDE1ABCDE2": 5,
            },
            "10fluc": {
                "A1A2": 1,
                "AB1AB2": 2,
                "ABC1ABC2": 3,
                "ABCD1ABCD2": 4,
                "ABCDE1ABCDE2": 5,
            },
        }
        adjust = adjust_table[self.fluc.split("_")[1]][self.alloy]
        # 產生A
        left_a = self.concentration[0][0]
        right_a = self.concentration[1][0]
        position = np.linspace(-adjust / 2, adjust / 2, layer_num)
        conc_gradient_a = 0.5 * (left_a + right_a) + 0.5 * (
            right_a - left_a
        ) * special.erf(position)
        # 產生B
        left_b = self.concentration[0][-1]
        right_b = self.concentration[1][-1]
        conc_gradient_b = 0.5 * (left_b + right_b) + 0.5 * (
            right_b - left_b
        ) * special.erf(position)

        return list(zip(conc_gradient_a, conc_gradient_b))

    def fccatomlist(self):
        total_alloy_atoms = self.couple_length * self.plane_atom_num

        # left end of diffusion couple
        num_of_atoms_left = [
            round(total_alloy_atoms * element_concentration)
            for element_concentration in self.concentration[0]
        ]
        num_of_atoms_left[-1] = total_alloy_atoms - sum(num_of_atoms_left[0:-1])
        atom_list_left = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(self.elements[0], num_of_atoms_left)
            ]
        )

        # right end of diffusion couple
        num_of_atoms_right = [
            round(total_alloy_atoms * element_concentration)
            for element_concentration in self.concentration[1]
        ]
        num_of_atoms_right[-1] = total_alloy_atoms - sum(num_of_atoms_right[0:-1])
        atom_list_right = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(self.elements[1], num_of_atoms_right)
            ]
        )
        return atom_list_left, atom_list_right

    def fccdiffusioncouple(self, atom_list_left, atom_list_right, conc_gradient):
        # 重新排列原子
        copy_atom_list_left = np.copy(atom_list_left)
        copy_atom_list_right = np.copy(atom_list_right)
        np.random.shuffle(copy_atom_list_left)
        np.random.shuffle(copy_atom_list_right)

        diffusion_couple_array = np.concatenate(
            [copy_atom_list_left, copy_atom_list_right]
        )

        # 根據設定erf 調整
        half_layer_num = len(conc_gradient) // 2
        for i, layer in enumerate(range(-half_layer_num, half_layer_num)):

            modified_conc = self.concentration[0]
            modified_conc[0], modified_conc[-1] = (
                conc_gradient[i][0],
                conc_gradient[i][-1],
            )
            num_of_atoms = [
                round(self.plane_atom_num * element_concentration)
                for element_concentration in modified_conc
            ]
            num_of_atoms[-1] = self.plane_atom_num - sum(num_of_atoms[0:-1])
            modified_atom_list = np.concatenate(
                [
                    np.full(element_nums, element, dtype=int)
                    for element, element_nums in zip(self.elements[0], num_of_atoms)
                ]
            )
            np.random.shuffle(modified_atom_list)
            diffusion_couple_array[
                ((self.couple_length + layer) * self.plane_atom_num) : (
                    (self.couple_length + layer + 1) * self.plane_atom_num
                )
            ] = modified_atom_list

        return diffusion_couple_array

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

    def vacancy_create(self, lattice_array):
        """
        Create a vacancy in center of lattice site, and return both new lattice and index of vancancy position
        """
        lattice = np.copy(lattice_array)
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
        lattice_record = {}
        v_position_record = {}  # 紀錄空孔最終位置
        num_repetitions = 0
        atom_list_left, atom_list_right = self.fccatomlist()
        erfc = self.erf_concentration_profile(self.couple_length * 2)
        error = 0
        while num_repetitions <= self.repetitions:
            # 進行重複取樣
            walkers_record[num_repetitions] = {}
            v_position_record[num_repetitions] = {}
            fcc_lattice = self.fccdiffusioncouple(atom_list_left, atom_list_right, erfc)
            lattice_record[num_repetitions] = fcc_lattice
            success = 0
            while success <= self.numofvacancy:
                # 持續擺放vacancy
                print(f"Now in {num_repetitions} {success}")
                # Lattice construct and randomly place atom onto it.
                lattice, v_position_index, remove_atom_type = self.vacancy_create(
                    fcc_lattice
                )
                vacancy_record = np.empty((self.steps, 3), dtype=float)
                position_recording = []  # 紀錄空孔位置
                check_in_layer = np.empty((self.steps,), dtype=int)
                try:
                    for step in range(self.steps):
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

                        if (step + 1) % 10 == 0:
                            position_recording.append(v_position_index)
                        check_in_layer[step] = v_position_index
                    if self.check_bound(check_in_layer):
                        error += 1
                        print("out of layer")
                        continue
                except Exception as e:
                    error += 1
                    print(e)
                    continue

                if not self.check_bound(check_in_layer):
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

                    success += 1
            num_repetitions += 1
        vacancy_trace = pd.DataFrame(walkers_record)
        filtered_lattice_record = pd.DataFrame(lattice_record)
        v_position = pd.DataFrame(v_position_record)

        return vacancy_trace, filtered_lattice_record, v_position, error

    def save_data(
        self, parameter, vacancy_record, lattice_record, v_position_record, error
    ):
        """
        Save the data with DataFrame in form of json
        """
        currentdir = os.path.abspath(os.curdir)
        dirname = f"fcc_gradient_diffpro_data_size{parameter[-1]}_numofvacancy{parameter[-3]}_{parameter[1]}"
        # file_dir = r"/work1/u3284480/"
        dirpath = os.path.join(
            currentdir, "gradient", f"{self.crystalstructure}", dirname
        )
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
            fh.write(f"error:{error}")
        os.chdir(currentdir)

        return


class BCCInterdiffusionGradient:
    def __init__(
        self,
        crystalstructure,
        fluc,
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
        self.crystalstructure = crystalstructure
        self.fluc = fluc
        self.alloy = alloy
        self.numofvacancy = int(numofvacancy)
        self.exchangerate = exchangerate
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

    def erf_concentration_profile(self, layer_num):
        adjust_table = {
            "5fluc": {
                "A1A2": 0.5,
                "AB1AB2": 2,
                "ABC1ABC2": 3,
                "ABCD1ABCD2": 4,
                "ABCDE1ABCDE2": 5,
            },
            "10fluc": {
                "A1A2": 1,
                "AB1AB2": 2,
                "ABC1ABC2": 3,
                "ABCD1ABCD2": 4,
                "ABCDE1ABCDE2": 5,
            },
        }
        adjust = adjust_table[self.fluc.split("_")[1]][self.alloy]
        # 產生A
        left_a = self.concentration[0][0]
        right_a = self.concentration[1][0]
        position = np.linspace(-adjust / 2, adjust / 2, layer_num)
        conc_gradient_a = 0.5 * (left_a + right_a) + 0.5 * (
            right_a - left_a
        ) * special.erf(position)
        # 產生B
        left_b = self.concentration[0][-1]
        right_b = self.concentration[1][-1]
        conc_gradient_b = 0.5 * (left_b + right_b) + 0.5 * (
            right_b - left_b
        ) * special.erf(position)

        return list(zip(conc_gradient_a, conc_gradient_b))

    def bccatomlist(self):
        total_alloy_atoms = self.couple_length * self.plane_atom_num

        # left end of diffusion couple
        num_of_atoms_left = [
            round(total_alloy_atoms * element_concentration)
            for element_concentration in self.concentration[0]
        ]
        num_of_atoms_left[-1] = total_alloy_atoms - sum(num_of_atoms_left[0:-1])
        atom_list_left = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(self.elements[0], num_of_atoms_left)
            ]
        )

        # right end of diffusion couple
        num_of_atoms_right = [
            round(total_alloy_atoms * element_concentration)
            for element_concentration in self.concentration[1]
        ]
        num_of_atoms_right[-1] = total_alloy_atoms - sum(num_of_atoms_right[0:-1])
        atom_list_right = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(self.elements[1], num_of_atoms_right)
            ]
        )
        return atom_list_left, atom_list_right

    def bccdiffusioncouple(self, atom_list_left, atom_list_right, conc_gradient):
        # 重新排列原子
        copy_atom_list_left = np.copy(atom_list_left)
        copy_atom_list_right = np.copy(atom_list_right)
        np.random.shuffle(copy_atom_list_left)
        np.random.shuffle(copy_atom_list_right)

        diffusion_couple_array = np.concatenate(
            [copy_atom_list_left, copy_atom_list_right]
        )

        half_layer_num = len(conc_gradient) // 2
        for i, layer in enumerate(range(-half_layer_num, half_layer_num)):

            modified_conc = self.concentration[0]
            modified_conc[0], modified_conc[-1] = (
                conc_gradient[i][0],
                conc_gradient[i][-1],
            )
            num_of_atoms = [
                round(self.plane_atom_num * element_concentration)
                for element_concentration in modified_conc
            ]
            num_of_atoms[-1] = self.plane_atom_num - sum(num_of_atoms[0:-1])
            modified_atom_list = np.concatenate(
                [
                    np.full(element_nums, element, dtype=int)
                    for element, element_nums in zip(self.elements[0], num_of_atoms)
                ]
            )
            np.random.shuffle(modified_atom_list)
            diffusion_couple_array[
                ((self.couple_length + layer) * self.plane_atom_num) : (
                    (self.couple_length + layer + 1) * self.plane_atom_num
                )
            ] = modified_atom_list

        return diffusion_couple_array

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

    def vacancy_create(self, lattice_array):
        """
        Create a vacancy in center of lattice site, and return both new lattice and index of vancancy position
        """
        lattice = np.copy(lattice_array)
        center = int(
            self.couple_length * self.plane_atom_num - (self.plane_atom_num // 2)
        )
        remove_atom_type = lattice[center]
        lattice[center] = 0
        return lattice, center, remove_atom_type

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
        Calculate exchange rate for vacancy with 1-st shell atom, which
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
        num_repetitions = 0
        atom_list_left, atom_list_right = self.bccatomlist()
        erfc = self.erf_concentration_profile(self.couple_length * 2)
        edge_fail = 0
        boundary_fail = 0
        while num_repetitions <= self.repetitions:
            # 進行重複取樣
            walkers_record[num_repetitions] = {}
            v_position_record[num_repetitions] = {}
            bcc_lattice = self.bccdiffusioncouple(atom_list_left, atom_list_right, erfc)
            lattice_record[num_repetitions] = bcc_lattice
            success = 0
            while success <= self.numofvacancy:
                # 持續擺放vacancy
                print(f"Now in {num_repetitions} {success}")
                # Lattice construct and randomly place atom onto it.
                lattice, v_position_index, remove_atom_type = self.vacancy_create(
                    bcc_lattice
                )
                vacancy_record = np.empty((self.steps, 3), dtype=float)
                position_recording = []  # 紀錄空孔位置
                check_in_layer = np.empty((self.steps,), dtype=int)
                try:
                    for step in range(self.steps):
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

                        if (step + 1) % 10 == 0:
                            position_recording.append(v_position_index)
                        check_in_layer[step] = v_position_index
                    if self.check_bound(check_in_layer):
                        print("out of layer")
                        continue
                except IndexError as e1:
                    print(e1)
                    boundary_fail += 1
                    continue
                except AssertionError as e2:
                    print(e2)
                    edge_fail += 1
                    continue

                if not self.check_bound(check_in_layer):
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

                    success += 1
            num_repetitions += 1
        vacancy_trace = pd.DataFrame(walkers_record)
        filtered_lattice_record = pd.DataFrame(lattice_record)
        v_position = pd.DataFrame(v_position_record)
        error = (edge_fail, boundary_fail)
        return vacancy_trace, filtered_lattice_record, v_position, error

    def save_data(
        self, parameter, vacancy_record, lattice_record, v_position_record, error
    ):
        """
        Save the data with DataFrame in form of json
        """
        currentdir = os.path.abspath(os.curdir)
        dirname = f"bcc_gradient_diffpro_data_size{parameter[-1]}_numofvacancy{parameter[-3]}_{parameter[1]}"
        # file_dir = r"/work1/u3284480/"
        dirpath = os.path.join(
            currentdir, "gradient", f"{self.crystalstructure}", dirname
        )
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
            fh.write(f"error:{error}")
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
    scwalker = SCInterdiffusionGradient(
        "scLattice",
        fluc,
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
    fccwalker = FCCInterdiffusionGradient(
        "fccLattice",
        fluc,
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
    bccwalker = BCCInterdiffusionGradient(
        "bccLattice",
        fluc,
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
        [1, 1],
        50,
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
