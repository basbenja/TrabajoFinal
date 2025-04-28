# **Cosas que me faltan**

## **Marco Teórico**

#### 1. **Evaluación de Impacto**
Del PSM:
- Agregar en qué consiste la regresión logística?

#### 2. **Inteligencia Artificial**

#### 3. **Redes Neuronales Artificiales**

#### 4. **Redes Neuronales Convolucionales**

#### 5. **Redes Neuronales Recurrentes**


#### 6. **Series de Tiempo**
Agregar los diferentes elementos presentes en una serie de tiempo

Simulation involves multiple parameters set to create various scenarios. These parameters include the number of simulations (N\_simulations), the total sample size (Total\_Sample), the proportion of treated individuals (PorTreated), and the proportion of control individuals (PorControl). Additional parameters, such as autoregressive persistence associated with the outcome variables, are specified for the treated (phiT), controls (PhiC), and neither treated nor control (Nini) groups (phiNini) to model temporal dependencies in the data.

A crucial aspect of our simulation is the modeling of the treatment effect, where we simulate a nominal impact of the treatment on the outcome variable based on a proportion (ImpactProportional) and with a variabilitiy in the impact on each individual that depend on a standard deviation (STDImpact). Furthermore, we introduce scenarios with both linear and nonlinear (nonlinear) dependencies in treatment assignment, reflecting the complex dynamics often observed in observational studies.

To examine the network's ability to handle heterogeneous treatment effects, we simulated variability in impact by cohort (staggered enters). This heterogeneity allows us to test the robustness of the LSTM network in identifying appropriate controls in different subpopulations with different treatment effects.

The data generation process involves the creation of time series for each individual, incorporating fixed effects, autoregressive components, and treatment impacts as appropriate. This process is designed to closely replicate the generation of observational data in studies with nonrandomized treatment assignments.