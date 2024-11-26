import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, classification_report,
    fbeta_score, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay,
    roc_auc_score
)

def standard_metrics(y_true, y_pred, beta=1):
    return {
        'precision':      precision_score(y_true, y_pred),
        'recall':         recall_score(y_true, y_pred),
        'f1_score':       f1_score(y_true, y_pred),
        'accuracy':       accuracy_score(y_true, y_pred),
        f'f{beta}_score': fbeta_score(y_true, y_pred, beta=beta)
    }


def confusion_matrix_plot(y, y_pred):
    ConfusionMatrixDisplay.from_predictions(y, y_pred)
    plt.title("Matriz de confusión")
    plt.show()


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
    X_pos = X_pos[:, :, 1]
    # Calculate the mean of each feature
    features_mean = X_pos.mean(dim=0)
    return features_mean
