import matplotlib.pyplot as plt
plt.style.use('default')

from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, roc_curve, roc_auc_score
)

def plot_time_series(df, n, label, n_per_dep, color=None, ax=None):
    """
    Draws time series plots for a random subset of rows in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data.
    n (int): Number of rows to plot.
    label (str): Label for the plot title. Should be one of "tratados", "controles"
    or "nini".
    """
    random_rows = df.sample(n=n, random_state=42)
    n_time_steps = len(df.columns)
    time_steps = list(range(-n_time_steps, 0))  # Time steps leading up to t

    if not ax:
        fig, ax = plt.subplots(figsize=(11, 6))
    else:
        fig = ax.figure

    for _, row in random_rows.iterrows():
        ax.plot(time_steps, row.values, color=color if color else None)

    ax.set_xlabel("$t$ (relativo al inicio del tratamiento)", fontsize=12)
    ax.set_ylabel("$y(t)$", fontsize=12)
    ax.set_title(f"Individuos {label}", fontsize=14, fontweight='bold', pad=20)
    
    if n_per_dep > 0:
        ax.axvspan(-n_per_dep, -1, color="red", alpha=0.2)
    
    existing_vlines = [l.get_label() for l in ax.get_lines()]
    if "Inicio de tratamiento" not in existing_vlines:
        ax.axvline(x=0, color='r', linestyle='--', label="Inicio de tratamiento")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(range(-n_time_steps, 1, 5))

    return fig, ax


def confusion_matrix_plot(y_true, y_pred, normalize=None):
    fig, ax = plt.subplots()

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    tn, fp, fn, tp = cm.ravel().tolist()
    confusion_dict = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["NiNi", "Control"]
    )
    disp.plot(ax=ax, cmap='viridis')

    ax.set_title(
        "Matriz de confusión",
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Verdadero")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")
    return fig, ax, confusion_dict


def roc_curve_plot(y, y_pred_probs):
    fig, ax = plt.subplots(figsize=(8, 5))

    fpr, tpr, thresholds = roc_curve(y, y_pred_probs)
    area = roc_auc_score(y, y_pred_probs)

    ax.plot(
        fpr, tpr, color='blue', label=f"Área bajo la curva = {area:.2f}"
    )
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Ratio de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Ratio de Verdaderos Positivos", fontsize=12)
    ax.set_title(
        "Característica Operativa del Receptor (ROC)",
        fontsize=14, fontweight='bold', pad=20
    )
    ax.grid()
    ax.legend(loc='lower right')

    return fpr, tpr, thresholds, area, fig, ax


def epoch_vs_loss_plot(epoch_losses_train, epoch_losses_test):
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = len(epoch_losses_train)

    ax.plot(range(epochs), epoch_losses_train, label="Entrenamiento")
    ax.plot(range(epochs), epoch_losses_test, label="Test")
    ax.set_xlabel("Época", fontsize=12)
    ax.set_ylabel("Pérdida promedio", fontsize=12)
    ax.set_title(
        "Pérdida promedio por época en conjuntos de entrenamiento y de test",
        fontsize=14, fontweight='bold', pad=20
    )
    ax.legend()
    ax.grid(True)
    ax.set_xlim(1, epochs)
    ax.set_xticks(range(0, epochs+1, 10))

    return fig, ax


def epoch_vs_metric_plot(metric_name, epoch_metrics):
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = len(epoch_metrics)

    ax.plot(range(epochs), epoch_metrics, label="Entrenamiento")
    ax.set_xlabel("Época", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(
        f"{metric_name} por época en conjunto de test",
        fontsize=14, fontweight='bold', pad=20
    )
    ax.legend()
    ax.grid(True)
    ax.set_xlim(1, epochs)
    ax.set_xticks(range(0, epochs+1, 10))

    return fig, ax
