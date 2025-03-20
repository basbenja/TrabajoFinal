Lo que vamos tener son n secuencias de 4 pasos temporales con 2 features: una feature variable con el tiempo (que es la variable de la cual tenemos datos), y la otra estática (que será el inicio del programa). n es la cantidad de individuos con la que vamos a entrenar a la red.

En PyTorch, el input de la LSTM tiene que tener el formato `(batch_size, sequence_length, num_features)`. Entonces, en nuestro caso el input debería ser de tamaño `(n, 4, 2)`. Veamos un ejemplo de esto para tenerlo más claro:
```python
# Si tuvieramos 3 secuencias por ejemplo
n = 3
x = torch.zeros(n, 4, 2)
x
```

Definimos la red neuronal. Por ahora la vamos a hacer bien simple:
1. Un layer LSTM.
2. Un layer fully connected (denso) para hacer la clasificación.

El módulo LSTM de PyTorch toma 3 parámetros principales (después hay alguno más que son más finos y los podríamos ver pero vamos con los obligatorios):
* `input_size`: el número de features en el input `x`. En nuestro caso, van a ser 2.
* `hidden_size`: el número de features en el estado oculto `h`. Se refiere a la dimensionalidad de los vecores de estado oculto, que son usados internamente por la LSTM para almacenar información de la secuencia que está procesando.
* `n_layers`: el número de capas recurrentes.

Algunas notas sobre `hidden_size` (fuente: ChatGPT):
* Una número mayor le da a la LSTM mayor memoria para capturar patrones más complejos en los datos, pero también incrementa el número de parámetros del modelo.
* Determina el tamaño del output de la LSTM en cada paso. Esto quiere decir que si por ejemplo la última capa es una densa (fully connected), esta debería tener como input un vector del tamaño de `hidden_size`.
* Algunas elecciones:
 * Si el dataset es pequeño o simple, un `hidden_size` de 16 o 32 quizás sea suficiente.
 * Para datasets más grandes y complejos, uno de 64 o 128 puede andar.

Otro parámetro importante, que tiene que ver con el formato de los inputs y los outputs es `batch_first`: si está en `True`, entonces los tensores de input y output se proveen como `(batch, seq, feature)` en lugar de `(seq, batch, feature)`. Por defecto, está en `False`.

Explicación de la indexación en:
```python
out = self.fc(out[:, -1, :])
```

La capa LSTM arroja un output de dimensiones `(batch_size, seq_length, hidden_size)`. Para pasar por la capa densa y clasificar, un enfoque común es utilizar el estado oculto correspondiente al último paso de la secuencia como un "resumen" de toda la secuencia.