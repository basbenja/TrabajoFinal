import json
import numpy as np
import os
import pandas as pd
import re

from typing import Literal
from statsmodels.tsa.arima_process import arma_generate_sample

from constants import DATA_DIR
from data.helpers import gen_next_time_step, gen_time_series_with_trend

class DataGenerator():
    DF_COLUMNS = [
        'sim', 'id', 'inicio_prog', 't', 'y', 'y_cf', 'tratado', 'control'
    ]

    def __init__(self, params: dict) -> "DataGenerator":
        # Variable params
        self.params = params
        self._parse_params()

        # Fixed params
        self.ini = 200
        self.std_error = 5

        # Impact params
        self.phi_effect = 0.90
        self.proportional_effect = 0.05
        self.nominal_effect = self.mean_FE_control * self.proportional_effect
        self.effect_std = 0.05

        self.hetecohorte = 1

        self.group_number = self._next_group_number()
        self.group_path = os.path.join(DATA_DIR, f"group_{self.group_number}")
        if os.path.exists(self.group_path):
            raise Exception(f"{self.group_path} already exists")
        os.makedirs(self.group_path)

        self._save_params()

    def _next_group_number(self) -> int:
        """
        Determine the next available group number by checking existing folders
        in the data directory.
        """
        pattern = r'group_(\d+)'
        groups = [d for d in os.listdir(DATA_DIR) if re.match(pattern, d)]
        numbers = [int(re.search(pattern, g).group(1)) for g in groups]
        return max(numbers) + 1 if numbers else 1

    def _parse_params(self) -> None:
        """
        Parse the parameters dictionary and set instance variables accordingly.
        """
        p = self.params

        self.n_simulations = p["n_simulations"]

        self.n_total = p["n_total"]
        self.n_treated = p["n_treated"]
        self.n_control = p["n_control"]
        self.n_nini = self.n_total - (self.n_treated + self.n_control)

        self.T = p["T"]
        self.total_periods = self.T + p.get("ini", 200)
        self.first_tr_period = p["first_tr_period"]

        self.n_cohorts = p["n_cohorts"]
        self.n_per_dep = p["n_per_dep"]
        self.ups_max_count = p["ups_max_count"]

        if self.n_per_dep > self.first_tr_period:
            raise ValueError(
                "n_per_dep cannot be greater than first_tr_period."
            )

        # Autorregresive parameters
        self.phi_treated = p["phiT"]
        self.phi_control = p["phiC"]
        self.phi_nini    = p["phiNiNi"]

        # Fixed effects parameters
        self.mean_FE_treated = p["MeanFET"]
        self.mean_FE_control = p["MeanFEC"]
        self.mean_FE_nini    = p["MeanFEN"]

    def _save_params(self) -> None:
        """
        Save the parameters used for data generation into a JSON file in the
        group's directory.
        """
        path = os.path.join(self.group_path, f"params.json")
        params_copy = dict(self.params)
        params_copy["group_number"] = self.group_number
        with open(path, "w") as f:
            json.dump(params_copy, f, indent=4)

    def _init_simulation_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize empty arrays to hold the generated data for treated, control
        and nini groups.
        """
        treated = np.zeros(shape=(self.n_treated * self.T, len(self.DF_COLUMNS)))
        control = np.zeros(shape=(self.n_control * self.T, len(self.DF_COLUMNS)))
        nini    = np.zeros(shape=(self.n_nini    * self.T, len(self.DF_COLUMNS)))
        return treated, control, nini

    def _generate_group(self, label: Literal["tratados", "controles"], mean_TE: float):
        if label == "tratados":
            n_units, mean_FE, phi = self.n_treated, self.mean_FE_treated, self.phi_treated
        elif label == "controles":
            n_units, mean_FE, phi = self.n_control, self.mean_FE_control, self.phi_control

        # Amount of treated units per cohort
        if self.n_treated % self.n_cohorts == 0:
            n_treated_in_cohort = int(self.n_treated / self.n_cohorts)
        else:
            n_treated_in_cohort = (
                int(self.n_treated / self.n_cohorts)
                + (self.n_treated % self.n_cohorts)
            )

        tr_starts_indexes = []
        y = np.zeros((n_units, self.T))

        # A fixed effect for each unit (FEs: Fixed Effects)
        FEs = np.random.normal(0, 1, n_units) + mean_FE
        for cohort in range(self.n_cohorts):
            # Subtract 1 to turn it 0-indexed
            tr_start_index = (self.ini + self.first_tr_period + cohort) - 1
            for i in range(n_treated_in_cohort):
                index = (cohort * n_treated_in_cohort) + i
                if index >= n_units:
                    continue

                y_i = gen_time_series_with_trend(
                    steps=self.total_periods,
                    n_per_dep=self.n_per_dep,
                    treatment_start=tr_start_index,
                    ups_max_count=self.ups_max_count,
                    phi=phi,
                    mean_time_effects=mean_TE,
                    mean_fixed_effects=mean_FE,
                    fixed_effect_i=FEs[i],
                    std_error=self.std_error
                )
                tr_starts_indexes.append(tr_start_index - self.ini)
                y[index, :] = y_i[self.ini:]

        return y, tr_starts_indexes

    def _generate_nini(self, mean_TE: float):
        n_units, mean_FE, phi = self.n_nini, self.mean_FE_nini, self.phi_nini
        y = np.zeros(shape=(n_units, self.T))

        # A fixed effect for each unit (FEs: Fixed Effects)
        FEs = np.random.normal(0, 1, n_units) + mean_FE

        for i in range(n_units):
            y_i = np.zeros(self.total_periods)
            y_i[0] = (
                mean_FE + FEs[i] + mean_TE + np.random.normal(0,1) * self.std_error
            )
            for t in range(self.total_periods):
                y_i[t] = gen_next_time_step(
                    mean_FE + FEs[i] + mean_TE, phi, y_i[t-1], self.std_error
                )
            y[i,:] = y_i[self.ini:]

        return y

    def _generate_single_simulation(self, sim_number: int) -> pd.DataFrame:
        y_panel_treated, y_panel_control, y_panel_nini = self._init_simulation_arrays()

        # A time effect, one for each time step (TEs = Time Effects)
        TEs = np.random.normal(0, 1, self.T + self.ini)
        mean_TE = TEs.mean()

        # Generate groups
        y_treated, tr_starts = self._generate_group(label='tratados' , mean_TE=mean_TE)
        y_control, _         = self._generate_group(label='controles', mean_TE=mean_TE)
        y_nini               = self._generate_nini(mean_TE=mean_TE)

        # Apply treatment
        y_treated_cf = y_treated.copy()

        # Apply treatment only to the treated units
        for i in range(self.n_treated):
            tr_start_index = tr_starts[i]
            tr_length = self.T - tr_start_index
            if self.hetecohorte == 1:
                arparams = np.array([self.phi_treated, 0])
                maparams = np.array([0, 0])
                arparams = np.r_[1, -arparams]
                maparams = np.r_[1, maparams]
                impact = arma_generate_sample(
                    arparams, maparams, tr_length, burnin=5000
                ) + self.nominal_effect
                y_treated[i, tr_start_index:] += impact
            else:
                y_treated[i, tr_start_index:] += np.random.normal(
                    self.nominal_effect, self.effect_std, tr_length
                )

        # Reshape treated and control data
        for panel, dataset, label in [
            (y_panel_treated, y_treated, 'tratados'),
            (y_panel_control, y_control, 'controles')
        ]:
            n = len(dataset)
            panel[:, 0] = sim_number
            panel[:, 6] = int(label == 'tratados')
            panel[:, 7] = int(label == 'controles')

            ids = np.zeros((n*self.T, 1))
            tr_starts = np.zeros((n*self.T, 1))
            steps = np.zeros((n*self.T, 1))
            y = np.zeros((n*self.T, 1))
            y_cf = np.zeros((n*self.T, 1))
            for i in range(n):
                ids[(i*self.T):((i+1)*self.T)] = i if label == 'tratados' else i + self.n_treated
                tr_starts[(i*self.T):((i+1)*self.T)] = tr_starts[i]
                steps[(i*self.T):((i+1)*self.T)] = np.arange(self.T).reshape(self.T, 1)
                y[(i*self.T):((i+1)*self.T)] = np.reshape(dataset[i,:], (self.T, 1))
                y_cf[(i*self.T):((i+1)*self.T)] = (
                    np.reshape(y_treated_cf[i,:], (self.T, 1)) if label == 'tratados' else 0
                )

            panel[:, 1] = np.reshape(ids, (n*self.T,))
            panel[:, 2] = np.reshape(tr_starts, (n*self.T,))
            panel[:, 3] = np.reshape(steps, (n*self.T,))
            panel[:, 4] = np.reshape(y, (n*self.T,))
            panel[:, 5] = np.reshape(y_cf, (n*self.T,))

        # Reshape Nini data
        n = self.n_nini
        y_panel_nini[:, 0] = sim_number

        ids = np.zeros((n*self.T, 1))
        steps = np.zeros((n*self.T, 1))
        y = np.zeros((n*self.T, 1))
        for i in range(n):
            ids[(i*self.T):((i+1)*self.T)] = i + (self.n_treated + self.n_control)
            steps[(i*self.T):((i+1)*self.T)] = np.arange(self.T).reshape(self.T, 1)
            y[(i*self.T):((i+1)*self.T)] = np.reshape(y_nini[i,:], (self.T, 1))

        y_panel_nini[:, 1] = np.reshape(ids,   (n*self.T,))
        y_panel_nini[:, 3] = np.reshape(steps, (n*self.T,))
        y_panel_nini[:, 4] = np.reshape(y,     (n*self.T,))
        # Columns 2, 5, 6 and 7 are already filled with zeros

        df_treated = pd.DataFrame(y_panel_treated, columns=self.DF_COLUMNS)
        df_control = pd.DataFrame(y_panel_control, columns=self.DF_COLUMNS)
        df_nini    = pd.DataFrame(y_panel_nini,    columns=self.DF_COLUMNS)

        df = pd.concat([df_treated, df_control, df_nini])
        df['sim'] = df['sim'].astype(int)
        df['id'] = df['id'].astype(int)
        df['inicio_prog'] = df['inicio_prog'].astype(int)
        df['t'] = df['t'].astype(int)
        df['tratado'] = df['tratado'].astype(int)
        df['control'] = df['control'].astype(int)

        return df

    def generate(self):
        for sim in range(1, self.n_simulations+1):
            print(f"Generating simulation {sim} of {self.n_simulations}...")
            df = self._generate_single_simulation(sim)
            sim_path = os.path.join(self.group_path, f"Simulacion{sim}.dta")
            df.to_stata(sim_path, write_index=False)
