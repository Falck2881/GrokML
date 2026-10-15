import numpy as np
from src.grok_ml.methods import IGradientDescent
from src.grok_ml.models import DirectModel

class Line:
    def __init__(self, features: np.ndarray, labels: np.ndarray, descent: IGradientDescent):
        self._features = features
        self._labels = labels
        self._descent = descent

    def create(self) -> DirectModel:
        """
        Создаёт модель линейной регрессии и возвращает параметры (вес/смещение)
        :return: Параметры (вес/смещение)
        """
        return self._descent.gradient(self._features, self._labels)

    @staticmethod
    def prediction(weight: float | np.ndarray, feature_x: float | np.int64 | np.ndarray, offset_b: float) -> float | np.ndarray:
        """
        Вычисляет предсказание уравнением линейной регрессии
        :param weight: веса/вес
        :param feature_x: параметр/параметры веса
        :param offset_b: смещение
        :return: предсказание `y
        """
        if isinstance(weight, np.ndarray) and isinstance(feature_x, np.ndarray):
            return weight @ feature_x + offset_b

        return weight * feature_x + offset_b

