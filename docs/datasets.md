# **Conjuntos de datos**
Vamos a necesitar:
- Conjunto de entrenamiento
- Conjunto de validación: para la búsqueda de hiperparámetros, nos vamos a fijar las métricas obtenidas en este conjunto.
- Conjunto de test: para evaluar el rendimiento final del modelo.

En todo el conjunto tenemos 1000 individuos, separados en:
- 100 individuos de tipo 1 (etiqueta 1)
- 100 individuos de tipo 2 (etiqueta 1)
- 800 individuos de tipo 3 (etiqueta 0)

Vamos a hacer una partición 70-20-10 (podría ser eventualmente 70-15-15). O sea, vamos a tener:
- 700 individuos en el conjunto de entrenamiento
- 200 individuos en el conjunto de validación
- 100 individuos en el conjunto de test

En cada conjunto, tenemos que mantener la proporción de 0s y 1s.

## **Alternativa 1**
Lo que distingue a esta alternativa es que vamos a meter a todos los de tipo 1 en el conjunto de entrenamiento.

La idea detrás de esto es que la red aprenda **solamente** de los tratados de forma tal de después predecir los de control a partir de ellos.

Vamos a tener entonces:
|Conjunto|Tipo 1|Tipo 2|Tipo 3|Total|Porcentaje 0s-1s|
|---|---|---|---|---|---|
|Entrenamiento|100|0|600|700|%86-%14|
|Validación|0|67|133|200|%66.5-%33.5|
|Test|0|33|67|100|%67-%33|

**Ventajas**:
- El modelo solamente aprende de los tratados para predecir los de control, que es lo que más nos interesa.

**Desventajas**:
- Los conjuntos no están igualmente balanceados.

## **Alternativa 2**
En el conjunto de entrenamiento tenemos solamente individuos de tipo 1, pero no todos, y el resto de tipo 3.

En el de validación metemos algunos de tipo 1 y otros de tipo 2, y el resto de tipo 3.

En el de test, metemos de tipo 2 y 3.

Vamos a tener entonces:
|Conjunto|Tipo 1|Tipo 2|Tipo 3|Total|Porcentaje 0s-1s|
|---|---|---|---|---|---|
|Entrenamiento|80|0|620|700|%89-%11|
|Validación|20|60|120|200|%60-%40|
|Test|0|40|60|100|%60-%40|

**Ventajas**:
- El modelo solamente aprende de los tratados para predecir los de control, que es lo que más nos interesa.
- En la búsqueda de hiperparámetros metemos algunos de los tratados.

**Desventajas**:
- Los conjuntos no están igualmente balanceados (peor que en la alternativa 1).

## **Alternativa 3**
|Conjunto|Tipo 1|Tipo 2|Tipo 3|Total|Porcentaje 0s-1s|
|---|---|---|---|---|---|
|Entrenamiento|80|0|300|380|%79-%21|
|Validación|20|0|100|120|%83-%17|
|Test|0|100|400|500|%80-%20|