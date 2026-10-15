import numpy as np
import random as rd
from abc import ABC, abstractmethod
from src.grok_ml.models import DirectModel
from src.grok_ml.errors_func import rmse


class IGradientDescent(ABC):
    """
    Градиентный спуск
    """
    @abstractmethod
    def gradient(self, features: np.ndarray, labels: np.ndarray) -> DirectModel:
        """
            Градиентный спуск
            :param features: Параметры для веса
            :param labels: метки
            :return: Параметры (вес/смещение)
        """
        pass


class StochasticGradientDescent(IGradientDescent):
    """
        Стохастический градиентный спуск
    """

    def __init__(self, learning_step: float = 0.01, iterations: int = 10000):
        self.__learning_step = learning_step
        self.__iterations = iterations

    def gradient(self, features: np.ndarray, labels: np.ndarray) -> DirectModel:
        from src.grok_ml.ml_models.regressive.line import Line
        weight: float = rd.random()
        bias: float = rd.random()
        epochs_errors: list[float] = []
        """
        Погрешность эпох
        """
        errors: list[float] = []
        """
        Погрешности 
        """
        endEpoch: int = 0
        """
        Конец эпохи
        """

        for _ in range(self.__iterations):
            i = rd.randint(0, len(features) - 1)
            feature_x = features[i]
            label_y = labels[i]
            predication_y = Line.prediction(weight, feature_x, bias)
            weight += self.__learning_step * feature_x * (label_y - predication_y)
            bias += self.__learning_step * (label_y - predication_y)

            if endEpoch == len(labels):
                print(f"endEpoch {endEpoch} - labels: {len(labels)}")
                all_predications = Line.prediction(weight, features, bias)
                errors.append(rmse(all_predications, labels))
                epochs_errors.append(len(errors))
                endEpoch = 0

            endEpoch += 1

        direction: DirectModel = DirectModel(errors=errors, epochs_errors=epochs_errors, weight=weight, bias=bias)
        return direction


class BatchGradientDescent(IGradientDescent):
    """
    Пакетный градиентный спуск
    """

    def __init__(self, learning_step: float = 0.01):
        self.__learning_step = learning_step

    def gradient(self, features: np.ndarray, labels: np.ndarray) -> DirectModel:
        from src.grok_ml.ml_models.regressive.line import Line
        weight: float = rd.random()
        bias: float = rd.random()
        """
        Производная по смещению
        """
        epochs_errors: list[float] = []
        """
        Погрешность эпох
        """
        errors: list[float] = []
        """
        Погрешности 
        """
        iteration: int = 500

        for iter in range(iteration):
            derivative_bias = 0.0
            derivative_weight = 0.0
            for i in range(len(labels)):
                feature_x = features[i]
                label_y = labels[i]
                predication_y = Line.prediction(weight, feature_x, bias)
                derivative_weight += -2 * feature_x * (label_y - predication_y)
                derivative_bias += -2 * (label_y - predication_y)

            weight -= self.__learning_step * (derivative_weight / len(labels))
            bias -= self.__learning_step * (derivative_bias / len(labels))

            all_predications = Line.prediction(weight, features, bias)
            errors.append(rmse(all_predications, labels))
            epochs_errors.append(iter)

        direction: DirectModel = DirectModel(errors=errors, epochs_errors=epochs_errors, weight=weight, bias=bias)
        return direction
