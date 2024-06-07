import os, time
import numpy as np
import pandas as pd
import multiprocessing as mp


class FCCRandomwalk:
    def __init__(
        self,
        alloy,
        elements,
        concentration,
        migration_barrier_mu,
        migration_barrier_sigma,
        temperature,
        steps=100000,
        realization=100,
        lattice_size=1000,
    ):
        self.alloy = alloy
        self.steps = int(steps)
        self.realization = int(realization)
        self.temp = int(temperature)
        self.const = 8.617 * 1e-5 * self.temp  # kT
        self.elements = int(elements)
        self.lattice_size = int(lattice_size)
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
        self.migration_barrier_mu = migration_barrier_mu
        self.migration_barrier_sigma = migration_barrier_sigma
        self.boundary = self.boundary_make()

    def fccatom(self):
        """
        Generate randomly distributed atom list in order to put into lattice site based on projection (111), and according to # of element,
        it will return [1,2,3,4,5] for each type of element kind.

        Paremeter
        ------------
        lattice size : sysytem size
        element : # of element
        """
        total_atoms = (2 * self.lattice_size + 1) ** 3

        num_of_atoms = [
            round(total_atoms * element_concentration)
            for element_concentration in self.concentration
        ]
        num_of_atoms[-1] = total_atoms - sum(num_of_atoms[0:-1])
        elements = [1, 2, 3, 4, 5]
        atom_list = np.concatenate(
            [
                np.full(element_nums, element, dtype=int)
                for element, element_nums in zip(elements, num_of_atoms)
            ]
        )
        np.random.shuffle(atom_list)
        return atom_list

    def vacancy_create(self, lattice):
        """
        Create a vacancy in center of lattice site, and return both new lattice and index of vancancy position
        """
        center = int(
            self.lattice_size * (2 * self.lattice_size + 1) ** 2
            + self.lattice_size * (2 * self.lattice_size + 1)
            + self.lattice_size
        )
        lattice[center] = 0

        return lattice, center

    def boundary_make(self):
        edge_atom_nums = 2 * self.lattice_size + 1
        back_boundary = np.arange(edge_atom_nums**2)
        front_boundary = np.arange(
            2 * self.lattice_size * edge_atom_nums**2, edge_atom_nums**3
        )

        right_boundary = np.concatenate(
            [
                np.arange(
                    i * (edge_atom_nums**2) - edge_atom_nums,
                    i * (edge_atom_nums**2),
                )
                for i in range(1, edge_atom_nums)
            ]
        )

        left_boundary = np.concatenate(
            [
                np.arange(
                    i * (edge_atom_nums**2),
                    i * (edge_atom_nums**2) + edge_atom_nums,
                )
                for i in range(1, edge_atom_nums)
            ]
        )

        up_boundary = np.arange(
            2 * self.lattice_size, edge_atom_nums**3, edge_atom_nums
        )
        down_boundary = np.arange(
            1 * (edge_atom_nums**2),
            edge_atom_nums**3,
            edge_atom_nums,
        )

        boundary = np.concatenate(
            (
                back_boundary,
                front_boundary,
                right_boundary,
                left_boundary,
                up_boundary,
                down_boundary,
            )
        )
        unique_boundary = np.unique(boundary)

        return unique_boundary

    def find_neighbor(self, lattice, v_position_index):
        """
        Find first shell of neighbor atom around vacancy of
        up-shell (up-right, up-left, down)
        same shell (right, up-right, up-left, left, down-left, down-right) and
        down shell (up, down-left, down-right)
        """
        edge_atom_nums = 2 * self.lattice_size + 1
        shell_index = v_position_index // edge_atom_nums**2
        row_index = (
            v_position_index - shell_index * (edge_atom_nums**2)
        ) // edge_atom_nums
        shell = shell_index % 3
        if shell == 1:
            if row_index % 2 == 0:
                neighbor_index = [
                    v_position_index - edge_atom_nums**2 + 1,
                    v_position_index - edge_atom_nums**2,
                    v_position_index - edge_atom_nums**2 - edge_atom_nums,
                    v_position_index + 1,  # right
                    v_position_index + edge_atom_nums,  # up-right
                    v_position_index + edge_atom_nums - 1,  # up-left
                    v_position_index - 1,  # left
                    v_position_index - edge_atom_nums - 1,  # down-left
                    v_position_index - edge_atom_nums,  # down-right
                    v_position_index + (edge_atom_nums**2),  # down shell up
                    v_position_index
                    + edge_atom_nums**2
                    - edge_atom_nums
                    - 1,  # down shell down-left
                    v_position_index + edge_atom_nums**2 - edge_atom_nums,
                ]
            else:
                neighbor_index = [
                    v_position_index - edge_atom_nums**2 + 1,
                    v_position_index - edge_atom_nums**2,
                    v_position_index - edge_atom_nums**2 - edge_atom_nums + 1,
                    v_position_index + 1,  # right
                    v_position_index + edge_atom_nums + 1,  # up-right
                    v_position_index + edge_atom_nums,  # up-left
                    v_position_index - 1,  # left
                    v_position_index - edge_atom_nums,  # down-left
                    v_position_index - edge_atom_nums + 1,
                    v_position_index + (edge_atom_nums**2),  # down shell up
                    v_position_index
                    + edge_atom_nums**2
                    - edge_atom_nums,  # down shell down-left
                    v_position_index + edge_atom_nums**2 - edge_atom_nums + 1,
                ]

                neighbor = [lattice[index] for index in neighbor_index]
        elif shell == 0:
            if row_index % 2 == 0:
                neighbor_index = [
                    v_position_index - edge_atom_nums**2,
                    v_position_index - edge_atom_nums**2 - 1,
                    v_position_index - edge_atom_nums**2 - edge_atom_nums - 1,
                    v_position_index + 1,  # right
                    v_position_index + edge_atom_nums,  # up-right
                    v_position_index + edge_atom_nums - 1,  # up-left
                    v_position_index - 1,  # left
                    v_position_index - edge_atom_nums - 1,  # down-left
                    v_position_index - edge_atom_nums,  # down-right
                    v_position_index
                    + (edge_atom_nums**2)
                    + edge_atom_nums
                    - 1,  # down shell up
                    v_position_index + edge_atom_nums**2 - 1,  # down shell down-left
                    v_position_index + edge_atom_nums**2,
                ]
            else:
                neighbor_index = [
                    v_position_index - edge_atom_nums**2,
                    v_position_index - edge_atom_nums**2 - 1,
                    v_position_index - edge_atom_nums**2 - edge_atom_nums,
                    v_position_index + 1,  # right
                    v_position_index + edge_atom_nums + 1,  # up-right
                    v_position_index + edge_atom_nums,  # up-left
                    v_position_index - 1,  # left
                    v_position_index - edge_atom_nums,  # down-left
                    v_position_index - edge_atom_nums + 1,
                    v_position_index
                    + (edge_atom_nums**2)
                    + edge_atom_nums,  # down shell up
                    v_position_index + edge_atom_nums**2 - 1,  # down shell down-left
                    v_position_index + edge_atom_nums**2,
                ]

        else:
            if row_index % 2 == 0:
                neighbor_index = [
                    v_position_index - edge_atom_nums**2 + edge_atom_nums,
                    v_position_index - edge_atom_nums**2 + edge_atom_nums - 1,
                    v_position_index - edge_atom_nums**2,
                    v_position_index + 1,  # right
                    v_position_index + edge_atom_nums,  # up-right
                    v_position_index + edge_atom_nums - 1,  # up-left
                    v_position_index - 1,  # left
                    v_position_index - edge_atom_nums - 1,  # down-left
                    v_position_index - edge_atom_nums,  # down-right
                    v_position_index
                    + (edge_atom_nums**2)
                    + edge_atom_nums,  # down shell up
                    v_position_index + edge_atom_nums**2,  # down shell down-left
                    v_position_index + edge_atom_nums**2 + 1,
                ]
            else:
                neighbor_index = [
                    v_position_index - edge_atom_nums**2 + edge_atom_nums + 1,
                    v_position_index - edge_atom_nums**2 + edge_atom_nums,
                    v_position_index - edge_atom_nums**2,
                    v_position_index + 1,  # right
                    v_position_index + edge_atom_nums + 1,  # up-right
                    v_position_index + edge_atom_nums,  # up-left
                    v_position_index - 1,  # left
                    v_position_index - edge_atom_nums,  # down-left
                    v_position_index - edge_atom_nums + 1,
                    v_position_index
                    + (edge_atom_nums**2)
                    + edge_atom_nums
                    + 1,  # down shell up
                    v_position_index + edge_atom_nums**2,  # down shell down-left
                    v_position_index + edge_atom_nums**2 + 1,
                ]
        neighbor = [lattice[index] for index in neighbor_index]

        return neighbor, neighbor_index

    def check_bound(self, position_index):
        return np.any(np.isin(position_index, self.boundary))

    def prob(self, neighbor):
        """
        Calculate exchange probability for vacancy with 1-st shell atom, which
        is base on migration barrier randomly drawn from distribution.
        """
        const = self.const  # kT
        prefactor = 10**13
        migration_energy = np.array(
            [
                np.random.normal(
                    self.migration_barrier_mu[int(atom) - 1],
                    self.migration_barrier_sigma[int(atom) - 1],
                )
                for atom in neighbor
            ]
        )
        ex_prob = np.exp(-migration_energy / const)
        rate = prefactor * ex_prob
        total_rate = np.sum(rate)

        return rate, total_rate

    def cumlative_rates(self, rates):
        return [0] + [np.sum(rates[:i]) for i in range(1, len(rates) + 1)]

    def select_event(self, cumulative_rates):
        sample_rate = np.random.uniform(0, 1) * cumulative_rates[-1]
        for index in range(len(cumulative_rates) - 1):
            if (sample_rate >= cumulative_rates[index]) and (
                sample_rate < cumulative_rates[index + 1]
            ):
                return index

    def determine_exchange(self, neighbor_index, select_index):
        """
        Determine the exchange direction
        """
        chosen_index = neighbor_index[select_index]
        chosen_vector = self.directions[select_index]

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

    def clock(self, total_rate):
        systime = -(1 / np.sum(total_rate)) * np.log(np.random.random())
        return systime

    def random_walk(self):
        walkers_record = {}
        fcc_lattice = self.fccatom()
        success = 0
        while success < self.realization:
            print(f"NOW IN {success}")
            # Lattice construct and randomly place atom onto it.
            np.random.shuffle(fcc_lattice)
            lattice, v_position_index = self.vacancy_create(fcc_lattice)
            vacancy_record = np.empty((self.steps, 3), dtype=float)
            sys_time_recording = np.empty((self.steps,))
            systime = 0
            boundary = False
            position_record = []
            try:
                for step in range(self.steps):
                    neighbor, neighbor_index = self.find_neighbor(
                        lattice, v_position_index
                    )
                    rates, total_rate = self.prob(neighbor)
                    cumulativerates = self.cumlative_rates(rates)
                    selectindex = self.select_event(cumulativerates)
                    # Determine the exchange diretion
                    v_after_index, v_movement_vector = self.determine_exchange(
                        neighbor_index, selectindex
                    )
                    # Perform the vacancy exchangement with chosen atom and direciton
                    lattice, v_position_index = self.perform_exchnage(
                        lattice, v_position_index, v_after_index
                    )
                    # Record the exchange moment
                    vacancy_record = self.record_exchange(
                        vacancy_record, step, v_movement_vector
                    )
                    # Advanced systime based on total rate
                    time_advanced = self.clock(total_rate)
                    systime += time_advanced
                    sys_time_recording[step] = systime

                    # Record vacancy index to check if reaching boundary
                    position_record.append(v_position_index)
                    if (step + 1) % 10000 == 0:
                        boundary = self.check_bound(position_record)
                        if boundary:
                            break
                        else:
                            position_record = []
            except Exception as e:
                continue
            if not boundary:
                vacancy_trace = np.cumsum(vacancy_record, axis=0)
                walkers_record["x" + str(success)] = vacancy_trace[:, 0]
                walkers_record["y" + str(success)] = vacancy_trace[:, 1]
                walkers_record["z" + str(success)] = vacancy_trace[:, 2]
                walkers_record["SD" + str(success)] = np.sum(
                    np.square(vacancy_trace), axis=1
                )
                walkers_record["Time" + str(success)] = sys_time_recording
                success += 1
            else:
                continue

        vacancy_trace = pd.DataFrame(walkers_record)

        return vacancy_trace

    def save_data(self, vacancy_record):
        """
        Save the data with DataFrame in form of json
        """
        vacancy_record.to_json(
            f"Vacancy trace of {self.alloy} with {(self.steps)} under migration barrier mean {tuple(self.migration_barrier_mu)} & std{tuple(self.migration_barrier_sigma)}.json.gz",
            compression="gzip",
        )

        return
