# Trabajo Final
Facultad de Matemática, Astronomía, Física y Computación - Universidad Nacional de Córdoba - 2024

Trabajo Final de Grado de la Licenciatura en Ciencias de la Computación

**Autor:** Benjamín Bas Peralta

**Directores:** Dr. Martín Domínguez y Mgter. David Giuliodori

## Servidor de `mlflow`
```bash
# Activar un entorno virtual en donde esté mlflow instalado
pyenv activate TrabajoFinal

# Correr con nohup y con > mlflow.log 2>&1 & para correrlo en background y no ver
# el log
nohup mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri ./mlflow-storage/mlruns/ --artifacts-destination ./mlflow-storage/mlartifacts/ > mlflow.log 2>&1 &
```

## Dashboard de optuna
```bash
# Activar un entorno virtual en donde esté optuna-dashboard instalado
pyenv activate TrabajoFinal

# Correr con nohup y con > optuna.log 2>&1 & para correrlo en background y no ver
# el log
nohup optuna-dashboard --port 8081 sqlite:///optuna_db.sqlite3  > optuna.log 2>&1 &
```

## Forwardear puertos
```bash
ssh -L 8080:localhost:8080 -L 8081:localhosto:8081 <user>@<server>
```