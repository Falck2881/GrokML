from dataclasses import dataclass

@dataclass(frozen=True)
class DirectModel:
     errors: list[float]
     """
     Массив погрешностей
     """
     epochs_errors: list[float]
     """
     Погрешность эпох
     """
     weight: float
     """
     Вес
     """
     bias: float
     """
     Смещение
     """