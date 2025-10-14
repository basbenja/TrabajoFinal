import json
import numpy as np
import os
import pandas as pd
import re

from constants import DATA_DIR
from data.helpers import gen_time_series_with_trend

COLUMNS = ['sim', 'id', 'inicio_prog', 't', 'y', 'y_cf', 'tratado', 'control']

class DataGenerator():
    def __init__(self, params):
        # Variable params
        self.params = params
        self._parse_params()

        # Fixed params
        self.ini = 200
        self.std_error = 5

        # Impact params
        self.proportional_effect = 0.05
        self.nominal_effect = self.mean_FE_control * self.proportional_effect
        self.effect_std = 0.05

        self.group_number = self._next_group_number()
        self.group_path = os.path.join(DATA_DIR, f"Grupo{self.group_number}")
        if os.path.exists(self.group_path):
            raise Exception(f"{self.group_path} already exists")

        self._save_params()

    def _next_group_number(self):
        pattern = r'Grupo(\d+)'
        groups = [d for d in os.listdir(self.base_dir) if re.match(pattern, d)]
        numbers = [int(re.search(pattern, g).group(1)) for g in groups]
        return max(numbers) + 1 if numbers else 1

    def _parse_params(self):
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

    def _save_params(self):
        path = os.path.join(self.group_path, f"params.json")
        params_copy = dict(self.params)
        params_copy["group_number"] = self.group_number
        with open(path, "w") as f:
            json.dump(params_copy, f, indent=4)

    def _init_simulation_arrays(self):
        treated = np.zeros(shape=(self.n_treated * self.T, len(COLUMNS)))
        control = np.zeros(shape=(self.n_control * self.T, len(COLUMNS)))
        nini    = np.zeros(shape=(self.n_nini    * self.T, len(COLUMNS)))
        return treated, control, nini

    def _generate_group(self, label, mean_TE):
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

        # A fixed effect for each unit
        EfectoFijoT = np.random.normal(0, 1, n_units) + mean_FE
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
                    mean_fixed_effects=mean_FE,
                    std_error=self.std_error,
                    fixed_effect_i=EfectoFijoT[i],
                    mean_temp_effects=mean_TE,
                    phi=phi,
                    treatment_start=tr_start_index,
                    ups_max_count=self.ups_max_count
                )
                tr_starts_indexes.append(tr_start_index - self.ini)
                y[index, :] = y_i[self.ini:]

        return y, tr_starts_indexes

    def _generate_nini():
        y_nini = np.zeros(shape=(n_nini, T))

        # Generate the data for the nini group
        EfectoFijoNiNi = np.random.normal(0, 1, n_nini) + MeanFEN

        for i in range(n_nini):
            y_i = np.zeros(total_periods)
            y_i[0] = (
                MeanFEN + EfectoFijoNiNi[i] + MeanTE + np.random.normal(0,1) * StdErrorSerie
            )
            for t in range(total_periods):
                y_i[t] = gen_next_time_step(
                    MeanFEN + EfectoFijoNiNi[i] + MeanTE, phiNiNi, y_i[t-1], StdErrorSerie
                )
            y_nini[i,:] = y_i[ini:]

        # Reshape the data from horizontal to vertical format with extra columns
        print(f"\tCambiando formato de datos para NiNi...")
        n = n_nini
        YPanelNiNi[:, 0] = sim + 1

        ids = np.zeros((n*T, 1))
        steps = np.zeros((n*T, 1))
        y = np.zeros((n*T, 1))
        for i in range(n):
            ids[(i*T):((i+1)*T)] = i + (n_treated + n_control)
            steps[(i*T):((i+1)*T)] = np.arange(T).reshape(T, 1)
            y[(i*T):((i+1)*T)] = np.reshape(y_nini[i,:], (T, 1))

        YPanelNiNi[:, 1] = np.reshape(ids, (n*T,))
        YPanelNiNi[:, 3] = np.reshape(steps, (n*T,))
        YPanelNiNi[:, 4] = np.reshape(y, (n*T,))
        # Columns 2, 5, 6 and 7 are already filled with zeros

    def _generate_single_simulation(self):
        y_panel_treated, y_panel_control, y_panel_nini = self._init_simulation_arrays()

        # A time effect, one for each time step
        TE = np.random.normal(0, 1, self.T + self.ini)
        mean_TE = TE.mean()

        # Generate groups
        y_treated, tr_starts = self._generate_group(label='tratados' , mean_TE=mean_TE)
        y_control, _         = self._generate_group(label='controles', mean_TE=mean_TE)

        # Apply treatment
        y_treated_cf = y_treated.copy()
        y_treated = self._apply_treatment(y_treated, tr_starts)

        # Apply treatment only to the treated units
        for i in range(n_treated):
            tr_start_index = tr_starts_indexes[i]
            tr_length = T - tr_start_index
            if hetecohorte == 1:
                arparams = np.array([phiTra, 0])
                maparams = np.array([0, 0])
                arparams = np.r_[1, -arparams]
                maparams = np.r_[1, maparams]
                impact = arma_generate_sample(
                    arparams, maparams, tr_length, burnin=5000
                ) + ImpactoNominal
                y_treated[i, tr_start_index:] += impact
            else:
                TraCondicion = np.array(tr_starts_indexes)
                y_treated[i, tr_start_index:] += np.random.normal(
                    ImpactoNominal, STDImpacto, tr_length
                )
                TraCondicion = np.array(tr_starts_indexes)

        # Reshape the data from horizontal to vertical format with extra columns
        for panel, dataset, label in [
            (YPanelTreated, y_treated, 'Tratados'),
            (YPanelControl, y_control, 'Controles')
        ]:
            n = len(dataset)
            panel[:, 0] = sim + 1
            panel[:, 6] = int(label == 'Tratados')
            panel[:, 7] = int(label == 'Controles')

            ids = np.zeros((n*T, 1))
            tr_starts = np.zeros((n*T, 1))
            steps = np.zeros((n*T, 1))
            y = np.zeros((n*T, 1))
            y_cf = np.zeros((n*T, 1))
            for i in range(n):
                ids[(i*T):((i+1)*T)] = i if label == 'Tratados' else i + n_treated
                tr_starts[(i*T):((i+1)*T)] = tr_starts_indexes[i]
                steps[(i*T):((i+1)*T)] = np.arange(T).reshape(T, 1)
                y[(i*T):((i+1)*T)] = np.reshape(dataset[i,:], (T, 1))
                y_cf[(i*T):((i+1)*T)] = (
                    np.reshape(y_counterfac[i,:], (T, 1)) if label == 'Tratados' else 0
                )

            panel[:, 1] = np.reshape(ids, (n*T,))
            panel[:, 2] = np.reshape(tr_starts, (n*T,))
            panel[:, 3] = np.reshape(steps, (n*T,))
            panel[:, 4] = np.reshape(y, (n*T,))
            panel[:, 5] = np.reshape(y_cf, (n*T,))


        print("\tGenerando DataFrames")
        df_treated = pd.DataFrame(YPanelTreated, columns=COLUMNS)
        df_control = pd.DataFrame(YPanelControl, columns=COLUMNS)
        df_nini = pd.DataFrame(YPanelNiNi, columns=COLUMNS)

        df = pd.concat([df_treated, df_control, df_nini])
        df['sim'] = df['sim'].astype(int)
        df['id'] = df['id'].astype(int)
        df['inicio_prog'] = df['inicio_prog'].astype(int)
        df['t'] = df['t'].astype(int)
        df['tratado'] = df['tratado'].astype(int)
        df['control'] = df['control'].astype(int)

    def generate(self):
        for sim in range(self.n_simulations):
            df = self._generate_single_simulation()
            sim_path = os.path.join(self.group_path, f"Simulacion{sim+1}.dta")
            df.to_stata(sim_path, write_index=False)
