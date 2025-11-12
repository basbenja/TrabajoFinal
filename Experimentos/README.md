# Aprendizaje Automático para la Selección de Grupos de Control en Evaluación de Impacto - Experimentos

## Configuración de proyecto
### Prerrequisitos
- Instalar [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

## Creación y activación de entorno virtual
1. Colocarse en esta carpeta.
2. Correr el siguiente comando, que automáticamente crea un entorno virtual e
instala las dependencias en el mismo.
```bash
uv sync
```
Esto creará una carpeta `.venv` en el directorio actual.
3. Una vez creado el entorno virtual, activarlo:
```bash
source .venv/bin/activate
```

### Creación de datasets
**NOTA**: los datasets generados quedarán almacenados en una carpeta `datasets` en
el directorio actual. Este directorio se puede cambiar en el archivo `constants.py`,
la constante llamada `DATA_DIR`.

Los datasets están organizados en **grupos**. Dentro de cada grupo, hay
**simulaciones**, que todas comparten los mismos parámetros. La idea de esto es
poder obtener resultados significativos para un determinado conjunto de parámetros.

1. Setear los valores deseados para los parámetros de la generación de datos en
el archivo `data_params.json`.
2. Pararse en el directorio actual ("Experimentos") y correr el siguiente comando:
```bash
python -m scripts.generate_data
```

## Entrenamiento de modelos
### Registro de experimentos: MLflow
Para registrar todo el proceso de entrenamiento y validación, se utiliza la
librería [**MLflow**](https://mlflow.org/docs/latest/ml/). Para levantar un
servidor **local** de MLflow, seguir las instrucciones en
[Self-Hosting MLflow](https://mlflow.org/docs/latest/self-hosting/).

Una vez que se haya levantado el servidor, hay que setear las constantes `HOST`
y `PORT` en el archivo `constants.py`.

### Optimización de hiperparámetros: Optuna
Para hacer la búsqueda de hiperparámetros, se utiliza la librería
[**Optuna**](https://optuna.readthedocs.io/en/stable/index.html).

Hay que setear la constante `SQLITE_DB_PATH` con el valor en donde estará el archivo
`.sqlite` que almacenará todos los registros de las corridas.

Para visualizar estos resultados, usar el [**dashboard de
Optuna**](https://github.com/optuna/optuna-dashboard):
```bash
optuna-dashboard <valor de la constante OPTUNA_STORAGE>
```

1. Setear los valores deseados para los parámetros de la generación de datos en
el archivo `train_params.json`. **NOTA**: si NO se levantó el servidor de MLflow,
setear el parámetro `log_to_mlflow` en `False`.