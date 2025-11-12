# Trabajo Final: Aprendizaje Automático para la Selección de Grupos de Control en Evaluación de Impacto - Experimentos

## Configuración de proyecto
### Prerrequisitos
- Instalar [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

### Creación y activación de entorno virtual
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

1. Setear los valores deseados para los parámetros de la generación de datos en
el archivo `data_params.json`.
2. Pararse en el directorio actual ("Experimentos") y correr el siguiente comando:
```bash
python -m scripts.generate_data
```
