# Trabajo Final
Facultad de Matemática, Astronomía, Física y Computación - Universidad Nacional de Córdoba - 2024

Trabajo Final de Grado de la Licenciatura en Ciencias de la Computación

**Autor:** Benjamín Bas Peralta

**Directores:** Dr. Martín Domínguez y Mgter. David Giuliodori

## Servidor de `mlflow`
```bash
# Activar un entorno virtual en donde esté mlflow instalado
pyenv activate tf

# Correr con nohup y con > mlflow.log 2>&1 & para correrlo en background y no ver
# el log
nohup mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri ./mlflow-storage/mlruns/ --artifacts-destination ./mlflow-storage/mlartifacts/ > mlflow.log 2>&1 &
```

## Dashboard de optuna
```bash
# Activar un entorno virtual en donde esté optuna-dashboard instalado
pyenv activate tf

# Correr con nohup y con > optuna.log 2>&1 & para correrlo en background y no ver
# el log
nohup optuna-dashboard --port 8081 sqlite:///optuna_db.sqlite3  > optuna.log 2>&1 &
```

## Forwardear puertos
```bash
ssh -L 8080:localhost:8080 -L 8081:localhost:8081 <user>@<server>
```

## Agregar nuevas arquitecturas de modelos
Por el momento, al momento de agregar una nueva arquitectura hay que hacer dos modificaciones:
1. Definir la arquitectura en un archivo aparte (hereda de `nn.Module`). Acá solamente
hay que definir los métodos `__init__` y `forward`.
2. Definir en la clase `DataPreprocessor` el preprocesamiento necesario de los
datos para convertirlos a un formato compatbile con la arquitectura.

# TODO
* Arquitectura y preprocesamiento en la misma clase?
* Agregar SMOTE a la clase `DataPreprocessor`?
* Instanciación automática del modelo en base al parámetro `MODEL_ARCH`.