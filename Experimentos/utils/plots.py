import matplotlib.pyplot as plt
plt.style.use('default')

from matplotlib.ticker import MultipleLocator
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, roc_curve, roc_auc_score
)

def plot_time_series(df, n, label):
    """
    Draws time series plots for a random subset of rows in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data.
    n (int): Number of rows to plot.
    label (str): Label for the plot title. Should be one of "tratados", "controles"
    or "nini".
    """
    random_rows = df.sample(n=n, random_state=42)
    steps = list(range(-len(df.columns), 0))  # Time steps leading up to t

    fig, ax = plt.subplots(figsize=(11, 5))

    for _, row in random_rows.iterrows():
        ax.plot(steps, row.values)

    ax.set_xlabel("$t$ (relativo al inicio del tratamiento)", fontsize=12)
    ax.set_ylabel("$y(t)$", fontsize=12)
    ax.set_title(f"Individuos {label}", fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='r', linestyle='--', label="Inicio de tratamiento")
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    return fig, ax


def confusion_matrix_plot(y_true, y_pred, normalize):
    fig, ax = plt.subplots()
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["NiNi", "Control"]
    )
    disp.plot(ax=ax, cmap='viridis')
    
    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Verdadero")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")
    return fig, ax


def roc_curve_plot(y, y_pred):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color='blue', lw=2, label='Area bajo la curva = %0.2f)' % roc_auc
    )
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Ratio de Falsos Positivos")
    plt.ylabel("Ratio de Verdaderos Positivos")
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def epochs_vs_loss_acc_plot(train_accs, train_avg_losses, test_accs, test_avg_losses):
    num_epochs = len(train_accs)
    num_epochs = range(1, num_epochs+1)

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(14, 6))

    axes[0].plot(num_epochs, train_accs, label="Train")
    axes[0].plot(num_epochs, test_accs, label="Test")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(num_epochs, train_avg_losses, label="Train")
    axes[1].plot(num_epochs, test_avg_losses, label="Test")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()

