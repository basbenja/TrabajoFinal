## **Configuración de los parámetros de las simulaciones**
---
* `n_simulations`: Número de simulaciones
* `n_total`: Número de elementos en la muestra (controles, tratados y Ni Ni)
* `control_pctg`: Porcentaje de individuos que van a ser tratados
* `n_control`, `n_treated`: Representan las cantidades de tratados y controles
* `n`: Número de elementos que son o bien tratado o controles. Se calcula como `n_control` + `n_treated`
* `T`: Número de períodos que son observados los individuos
* `first_tr_period`: Primer período de tratamiento (o único dependiendo de la cantiadad de cohortes=n_cohorts)

* `phiNini`: Persistencia auto-regresiva asociada a la variable resultado de los Nini
* `phiT`: Persistencia auto-regresiva asociada a la variable resultado de los tratados
* `phiC`: Persistencia auto-regresiva asociada a la variable resultado de los controles
* `meanFEN`: Media (nivel) del efecto fijo de los Nini
* `meanFET`: Media (nivel) del efecto fijo de los tratados
* `meanFEC`: Media (nivel) del efecto fijo de los controles
* `NivelN`: Media (nivel) de la variable resultado de los Nini sin el fijo
* `NivelT`: Media (nivel) de la variable resultado de los tratados sin el fijo
* `NivelC`: Media (nivel) de la variable resultado de los controles sin el fijo
* `ImpactoProporcional`: Porcentaje promedio del impacto tipo escalón sobre la la media del nivel de la variable resultado
* `STDImpacto`: Desviación estándar de la distribución del impacto.
* `dependence`: Si se simular dependence en la entrada al programa, si es 1 se genera con dependence el ingreso al tratamiento.
* `hetecohorte`: Heterogeneidad en el impacto por cohorte. Cambia la estructura dinámica del escalón. Puede ser aleatorio o mediante un proceso ARMA (caso donde la variable hetecohorte = 1), en este último caso, la dinámica del modelo ARMA cambia para cada cohorte.
* `n_per_dep`: Número de períodos de dependence para la participación en el tratamiento, pueden ser 4 ó 7.
* `nolineal`: Es 1 (uno) si se quiere simular una dependence de entrada al tratamiento que sea no lineal, es 0 (cero) si la dependence es lineal
* `ini`: cantidad de observaciones que se queman para que la dinámica del proceso auto-regresivo de la variable resultado esté ya en situación de estado estacionario
* `n_cohorts`: Número de cohortes
* `StdErrorSerie`: Desviación estándar del térmio de ruido de las series