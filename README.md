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

## Para usar `pystata`
`pystata` es un paquete que ya se instala junto con stata. Si stata se encuentra en
`/usr/local/stata17`, el paquete `pystata` está en `/usr/local/stata17/utilities/pystata`.

Leer las diferentes opciones para incluirlo en el entorno Python utilizado en
[pystata17 Installation](https://www.stata.com/python/pystata17/install.html).

Según la documentación de pystata, la forma más directa de interactuar con Stata en
una notebook de Jupyter, es usando las magics:
  - [The stata magic](https://www.stata.com/python/pystata19/notebook/Magic%20Commands1.html#):
    esta es la magic que justamente sirve para correr comandos Stata.
  - [The pystata magic](https://www.stata.com/python/pystata19/notebook/Magic%20Commands3.html):
    esta magic solamente sirve para configurar comportamientos/ajustes en la interacción
    entre Python y Stata.

# TODO
* Arquitectura y preprocesamiento en la misma clase?
* Agregar SMOTE a la clase `DataPreprocessor`?
* Instanciación automática del modelo en base al parámetro `MODEL_ARCH`.