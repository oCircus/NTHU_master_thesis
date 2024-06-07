import numpy as np
import pandas as pd

class BCCRandomwalk:
    def __init__(
        self,
        alloy,
        elements,
        concentration,
        migration_barrier_mu,
        migration_barrier_sigma,
        steps=100000,
        realization=100,
        lattice_size=1000,
    ):
        self.alloy = alloy
        self.steps = int(steps)
        self.realization = int(realization)
        self.const = 0.08625  # kT
        self.elements = int(elements)
        self.lattice_size = int(lattice_size)
        self.directions = np.array(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ]
        )
        self.concentration = concentration
        self.migration_barrier_mu = migration_barrier_mu
        self.migration_barrier_sigma = migration_barrier_sigma
        self.boundary = self.boundary_make()

    def bccatom(self):
        """
        Generate randomly distributed atom list in order to put into lattice site, according to # of element,
        it will return [1,2,3,4,5] for each type of element kind.

        Paremeter
        ------------
        lattice_size: # of atoms to put into, it should be 2N of trimake input value N ;

        element: # of element
        """
        total_atoms = (2 * self.lattice_size + 1) ** 3 + (2 * self.lattice_size) ** 3

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
        """
        Returns the boundary position index
        """
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
        # 扣除重複計算的boundary index
        unique_boundary = np.unique(boundary)

        return unique_boundary

    def find_neighbor(self, lattice, v_position_index):
        """
        Identify the atomic species of the first nearest neighbor to bcc, returning both neighbor atom and neighbor index.
        """
        # 找出空孔目前位置在體心或角落
        corner = (2 * self.lattice_size + 1) ** 3
        if v_position_index <= corner:
            x_index = v_position_index // ((2 * self.lattice_size + 1) ** 2)
            remaining_index = v_position_index % ((2 * self.lattice_size + 1) ** 2)
            y_index = remaining_index // (2 * self.lattice_size + 1)
            z_index = remaining_index % (2 * self.lattice_size + 1)

            body_initial_index = int((2 * self.lattice_size + 1) ** 3)
            neg_x_plane, pos_x_plane = x_index - 1, x_index
            neg_y_plane, pos_y_plane = y_index - 1, y_index
            neg_z_plane, pos_z_plane = z_index - 1, z_index
            a = (
                body_initial_index
                + neg_x_plane * (2 * self.lattice_size) ** 2
                + neg_y_plane * (2 * self.lattice_size)
                + neg_z_plane
            )
            b = (
                body_initial_index
                + neg_x_plane * (2 * self.lattice_size) ** 2
                + pos_y_plane * (2 * self.lattice_size)
                + neg_z_plane
            )
            c = (
                body_initial_index
                + pos_x_plane * (2 * self.lattice_size) ** 2
                + neg_y_plane * (2 * self.lattice_size)
                + neg_z_plane
            )
            d = (
                body_initial_index
                + pos_x_plane * (2 * self.lattice_size) ** 2
                + pos_y_plane * (2 * self.lattice_size)
                + neg_z_plane
            )
            neighbor = [
                lattice[a],
                lattice[a + 1],
                lattice[b],
                lattice[b + 1],
                lattice[c],
                lattice[c + 1],
                lattice[d],
                lattice[d + 1],
            ]
            neighbor_index = [a, a + 1, b, b + 1, c, c + 1, d, d + 1]
        else:
            v_position_index -= corner
            x_index = v_position_index // ((2 * self.lattice_size) ** 2)
            remaining_index = v_position_index % ((2 * self.lattice_size) ** 2)
            y_index = remaining_index // (2 * self.lattice_size)
            z_index = remaining_index % (2 * self.lattice_size)

            neg_x_plane, pos_x_plane = x_index, x_index + 1
            neg_y_plane, pos_y_plane = y_index, y_index + 1
            neg_z_plane, pos_z_plane = z_index, z_index + 1
            a = (
                neg_x_plane * (2 * self.lattice_size + 1) ** 2
                + neg_y_plane * (2 * self.lattice_size + 1)
                + neg_z_plane
            )
            b = (
                neg_x_plane * (2 * self.lattice_size + 1) ** 2
                + pos_y_plane * (2 * self.lattice_size + 1)
                + neg_z_plane
            )
            c = (
                pos_x_plane * (2 * self.lattice_size + 1) ** 2
                + neg_y_plane * (2 * self.lattice_size + 1)
                + neg_z_plane
            )
            d = (
                pos_x_plane * (2 * self.lattice_size + 1) ** 2
                + pos_y_plane * (2 * self.lattice_size + 1)
                + neg_z_plane
            )
            neighbor = [
                lattice[a],
                lattice[a + 1],
                lattice[b],
                lattice[b + 1],
                lattice[c],
                lattice[c + 1],
                lattice[d],
                lattice[d + 1],
            ]
            neighbor_index = [a, a + 1, b, b + 1, c, c + 1, d, d + 1]
        return neighbor, neighbor_index

    def check_bound(self, position_index):
        """
        Check whether vacancy reach boundary.
        """
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
        """
        Advance sysytime based on total rate and random number.
        """
        systime = -(1 / np.sum(total_rate)) * np.log(np.random.random())
        return systime

    def random_walk(self):
        walkers_record = {}
        bcc_lattice = self.bccatom()
        success = 0
        while success < self.realization:
            # Lattice construct and randomly place atom onto it.
            np.random.shuffle(bcc_lattice)
            lattice, v_position_index = self.vacancy_create(bcc_lattice)
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



