import os, json
import numpy as np
import pandas as pd
import multiprocessing as mp


class SCInterdiffusionMonoV:
    def __init__(
        self,
        alloy,
        elements,
        concentration,
        exchangeprobability,
        recordsteps,
        aspectratio=[4, 1],
        steps=100000,
        realization=100,
        lattice_size=1000,
    ):
        self.steps = int(steps)
        self.alloy = alloy
        self.realization = int(realization)
        self.recordsteps = recordsteps
        self.exchangeprobability = exchangeprobability
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
            self.interfaceboundary,
        ) = self.boundary_make()

    def scatomlist(self):
        alloy_atoms = self.couple_length * self.plane_atom_num
        # left end of diffusion couple
        num_of_atoms_left = [
            round(alloy_atoms * element_concentration)
            for element_concentration in self.concentration[0]
        ]
        num_of_atoms_left[-1] = alloy_atoms - sum(num_of_atoms_left[0:-1])
        atom_list_left = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(self.elements[0], num_of_atoms_left)
            ]
        )
        np.random.shuffle(atom_list_left)
        # right end of diffusion couple
        num_of_atoms_right = [
            round(alloy_atoms * element_concentration)
            for element_concentration in self.concentration[1]
        ]
        num_of_atoms_right[-1] = alloy_atoms - sum(num_of_atoms_right[0:-1])
        atom_list_right = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(self.elements[1], num_of_atoms_right)
            ]
        )
        np.random.shuffle(atom_list_right)

        return atom_list_left, atom_list_right

    def scdiffusioncouple(self, atom_list_left, atom_list_right):
        np.random.shuffle(atom_list_left)
        np.random.shuffle(atom_list_right)

        diffusion_couple_array = np.concatenate([atom_list_left, atom_list_right])
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

        interface_boundary = np.arange(
            self.plane_atom_num * self.couple_length - 5 * self.plane_atom_num,
            self.plane_atom_num * self.couple_length + 5 * self.plane_atom_num,
        )
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
            interface_boundary,
        )

    def check_bound(self, position_index):
        return np.any(np.isin(position_index, self.leftrightboundary))

    def check_in_layer(self, position_index):
        return np.any(np.isin(position_index, self.interfaceboundary))

    def vacancy_create(self, lattice):
        """
        Create a vacancy in center of lattice site, and return both new lattice and index of vancancy position
        """
        center = int(
            self.couple_length * self.plane_atom_num - (self.plane_atom_num // 2)
        )
        lattice[center] = 0

        return lattice, center

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
        """
        Calculate exchange probability for vacancy with 1-st shell atom, which
        is base on migration barrier randomly drawn from distribution.
        """
        ex_probability = np.array(
            [np.exp(-self.exchangeprobability[int(atom) - 1]) for atom in neighbor_atom]
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

    def random_walk(self):
        walkers_record = {}
        lattice_record = {}
        atom_list_left, atom_list_right = self.scatomlist()
        success = 0
        fail = 0
        interface_record = []
        v_position_record = {}
        while success < self.realization:
            # Lattice construct and randomly place atom onto it.
            sc_lattice = self.scdiffusioncouple(atom_list_left, atom_list_right)
            lattice_record[success] = {}
            lattice_record[success]["0"] = np.copy(sc_lattice).tolist()
            lattice, v_position_index = self.vacancy_create(sc_lattice)
            vacancy_record = np.empty((self.steps, 3), dtype=float)
            position_recording = np.empty((self.steps,), dtype=int)
            boundary = False
            interface = False
            position_record = []
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
                    ) = self.determine_exchange(neighbor_index, exchange_probability)
                    # Perform the vacancy exchangement with chosen atom and direciton
                    lattice, v_position_index = self.perform_exchnage(
                        lattice, v_position_index, v_after_index
                    )

                    # Record the exchange moment
                    vacancy_record = self.record_exchange(
                        vacancy_record, step, v_movement_vector
                    )
                    position_record.append(v_position_index)
                    position_recording[step] = v_position_index
                    if not interface:
                        interface_stat = self.check_in_layer(v_position_index)
                        if not interface_stat:
                            interface_record.append(step)
                            interface = True
                    if (step + 1) in self.recordsteps:
                        boundary = self.check_bound(position_record)
                        if boundary:
                            print("Reach SIDE boundary")
                            fail += 1
                            if success in lattice_record:
                                del lattice_record[success]
                            break
                        else:
                            position_record = []
                            if step not in lattice_record:
                                lattice_record[step] = {}
                            lattice_record[success][f"{step+1}"] = np.copy(
                                lattice
                            ).tolist()

            except Exception as e:
                fail += 1
                if success in lattice_record:
                    del lattice_record[success]
                continue
            if not boundary:
                vacancy_trace = np.cumsum(vacancy_record, axis=0)
                walkers_record["x" + str(success)] = vacancy_trace[:, 0]
                walkers_record["y" + str(success)] = vacancy_trace[:, 1]
                walkers_record["z" + str(success)] = vacancy_trace[:, 2]
                walkers_record["SD" + str(success)] = np.sum(
                    np.square(vacancy_trace), axis=1
                )
                v_position_record[success] = position_recording
                success += 1
            else:
                continue
        vacancy_trace = pd.DataFrame(walkers_record)
        filtered_lattice_record = {
            key: value for key, value in lattice_record.items() if value
        }
        v_position = pd.DataFrame(v_position_record)

        return vacancy_trace, filtered_lattice_record, v_position

    def save_data(self, parameter, vacancy_record, lattice_record, v_position_record):
        """
        Save the data with DataFrame in form of json
        """
        currentdir = os.path.abspath(os.curdir)
        dirname = (
            f"diffpro_data_size{parameter[-1]}_trials{parameter[-2]}_{parameter[1]}"
        )

        dirpath = os.path.join(currentdir, dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        os.chdir(dirpath)
        data_path = os.path.join(dirpath, f"{self.alloy}")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        os.chdir(data_path)
        lattice_record.to_json(
            f"Lattice record of {self.alloy} json.gz",
            compression="gzip",
        )
        vacancy_record.to_json(
            f"Vacancy trace of {self.alloy} with {(self.steps)} json.gz",
            compression="gzip",
        )
        v_position_record.to_json(
            f"Vacancy position of {self.alloy} with {(self.steps)} json.gz",
            compression="gzip",
        )
        with open(f"{self.alloy}_Parameter.txt", "w") as fh:
            parameter_name = [
                "alloy",
                "fluc",
                "elements",
                "concentration",
                "exchangeprobability",
                "recordsteps",
                "ratio",
                "steps",
                "trail",
                "lattice_size",
            ]
            for name, para in zip(parameter_name, parameter):
                fh.write(f"{name}:{para}\n")
        os.chdir(currentdir)

        return

