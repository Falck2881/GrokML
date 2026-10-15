import numpy as np


def rmse(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Вычисляет среднеквадратичную погрешность
    :param predictions: предсказания
    :param labels: метки
    :return: погрешность
    """
    quantity = len(labels)
    differences = np.subtract(labels, predictions)
    return np.sqrt(1/quantity * (np.dot(differences, differences)))