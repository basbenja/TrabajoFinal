**Terminología:**
- $VP$: Verdaderos Positivos. Es decir, individuos cuyo target era 1 y el modelo los predijo como 1.
- $VN$: Verdaderos Negativos. Es decir, individuos cuyo target era 0 y el modelo los predijo como 0.
- $FP$: Falsos Positivos. Es decir, individuos cuyo target era 0 pero el modelo los predijo como 1.
- $FN$: Falsos Negativos. Es decir, individuos cuyo target era 1 pero el modelo los predijo como 0.

**1. Accuracy**  
Es el ratio de instancias correctamente predichas (tanto positivas como negativas) sobre todos los individuos de la muestra:
$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

**2. Precisión**  
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

**3. Especificidad: Ratio de Verdaderos Negativos**  
Instancias negativas correctamente predichas sobre todas las instancias verdaderamente negativas.
$$
\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}
$$

**4. Recall: Ratio de Verdaderos Positivos**  
Instancias positivas correctamente predichas sobre todas las instancias verdaderamente positivas.
$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

**5. Puntaje F1**  
El puntaje F1 es la media armónica entre precisión y recall. Representa simétricamente a ambas métricas.
$$
\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**6. Puntaje F-Beta**  
Generaliza al puntaje F1 permitiendo asignar un peso al recall relativo a la precisión
$$
\text{F}_{\beta} = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}
$$

$\beta > 1$ enfatiza recall, and $\beta < 1$ enfatiza precisión.
