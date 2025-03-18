import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, fbeta_score,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

METRICS = {
    'precision': lambda y, y_pred: precision_score(y, y_pred, zero_division=0),
    'recall': lambda y, y_pred: recall_score(y, y_pred),
    'f1_score': lambda y, y_pred: f1_score(y, y_pred),
    'accuracy': lambda y, y_pred: accuracy_score(y, y_pred),
    'f_beta_score': lambda y, y_pred, beta: fbeta_score(y, y_pred, beta=beta),
    'avg_feats_diff': lambda X_valid, y_valid_pred, train_features_mean:
        avg_features_diffs(
            X_valid, y_valid_pred, train_features_mean
        )
}

def check_metrics(metrics, **kwargs):
    for metric in metrics:
        if metric == 'f_beta_score' and 'beta' not in kwargs:
            raise ValueError("You must provide the beta parameter for the f_beta_score metric")
        if metric not in METRICS:
            raise ValueError(f"Metric {metric} is not valid. Choose one of {METRICS.keys()}")


def compute_metrics(metrics, y_true, y_pred, **kwargs) -> dict[str, float]:
    # Scikit learn metrics require the labels to be numpy arrays in the CPU
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    metrics_dict = {metric: None for metric in metrics}
    for metric in metrics:
        if metric == 'avg_feats_diff':
            if 'X_valid' not in kwargs or 'train_features_mean' not in kwargs:
                raise ValueError(
                    "You must provide the X_valid and train_features_mean parameters "
                    "to calculate the avg_feats_diff metric"
                )
            metrics_dict[metric] = METRICS[metric](
                kwargs['X_valid'], y_pred, kwargs['train_features_mean']
            )
        elif metric == 'f_beta_score':
            if 'beta' not in kwargs:
                raise ValueError(
                    "You must provide the beta parameter for the f_beta_score metric"
                )
            metrics_dict[metric] = METRICS[metric](
                y_true, y_pred, beta=kwargs['beta']
            )
        else:
            metrics_dict[metric] = METRICS[metric](y_true, y_pred)
    return metrics_dict


def confusion_matrix_plot(y, y_pred, normalize='true'):
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, labels=["No Control", "Control"], normalize=normalize
    )
    plt.title("Matriz de confusión")
    fig = plt.gcf()
    return fig


def roc_curve_plot(y, y_pred):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
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


def get_features_mean(X, y):
    """
    Get the mean of the features for each period.
    """ 
    # Get the inputs whose output is 1
    X_pos = X[y == 1]
    # Keep only the temporal features. Each column represents a feature
    X_pos = X_pos[:, :, 0]
    # Calculate the mean of each feature
    features_mean = X_pos.mean(dim=0)
    return features_mean


def avg_features_diffs(X_valid, y_valid_pred, train_features_mean):
    valid_features_mean = get_features_mean(X_valid, y_valid_pred)
    differences_sum = sum(abs(train_features_mean - valid_features_mean))
    return differences_sum.item()