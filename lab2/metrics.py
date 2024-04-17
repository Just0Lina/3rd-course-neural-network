import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_score, recall_score


def count_accuracy(true_positive, true_negative, false_positive, false_negative) -> float:
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)


def precision(true_positive, false_positive) -> float:
    return true_positive / (true_positive + false_positive)


def recall(true_positive, false_negative) -> float:
    return true_positive / (true_positive + false_negative)


def f_score(percision, recall) -> float:
    return 2 * percision * recall / (percision + recall)


def count_metrics(true_positive, true_negative, false_positive, false_negative):
    print("accuracy: " + str(count_accuracy(true_positive, true_negative, false_positive, false_negative)))
    print("precision: " + str(precision(true_positive, false_positive)))
    print("recall: " + str(recall(true_positive, false_negative)))
    print("f-score: " + str(f_score(precision(true_positive, false_positive), recall(true_positive, false_negative))))


def count_metrics(true, predicted):
    # Вычисляем precision и recall для каждого класса
    precision = precision_score(true, predicted, average=None)
    recall = recall_score(true, predicted, average=None)

    # Выводим precision и recall для каждого класса
    for i in range(len(precision)):
        print("Class {}: Precision = {:.4f}, Recall = {:.4f}".format(i, precision[i], recall[i]))

    # Также можно вычислить средневзвешенное precision и recall
    weighted_precision = precision_score(true, predicted, average='weighted')
    weighted_recall = recall_score(true, predicted, average='weighted')
    print("Weighted Precision:", weighted_precision)
    print("Weighted Recall:", weighted_recall)


def compute_tpr_fpr(predictions, true_labels):
    thresholds = sorted(set(predictions), reverse=True)
    tprs = []
    fprs = []

    for threshold in thresholds:
        binarized_predictions = [1 if pred >= threshold else 0 for pred in predictions]

        tp = sum((p == 1 and t == 1) for p, t in zip(binarized_predictions, true_labels))
        fp = sum((p == 1 and t == 0) for p, t in zip(binarized_predictions, true_labels))
        tn = sum((p == 0 and t == 0) for p, t in zip(binarized_predictions, true_labels))
        fn = sum((p == 0 and t == 1) for p, t in zip(binarized_predictions, true_labels))

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        tprs.append(tpr)
        fprs.append(fpr)

    return tprs, fprs


def auc_manual(fpr, tpr):
    auc_score = 0
    n = len(fpr)
    for i in range(1, n):
        auc_score += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return auc_score


def custom_roc_curve(predictions, target):
    tpr, fpr = compute_tpr_fpr(predictions, target)
    roc_auc = auc_manual(fpr, tpr)
    print("AUC-ROC (Custom):", roc_auc[0])
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (Custom)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (Custom)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.show()


def sklearn_roc_curve(predictions, target):
    fpr, tpr, thresholds = roc_curve(target, predictions)

    # Вычисляем площадь под ROC-кривой (AUC)
    roc_auc = auc(fpr, tpr)
    print("AUC-ROC (SKlearn):", roc_auc)

    # Строим ROC-кривую
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (SKlearn)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (SKlearn)')
    plt.legend()
    plt.show()
