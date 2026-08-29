import pandas as pd

class CsvService:
    """
    Сервис CSV распаковывающий содержимое файла
    """
    def __init__(self, name_file: str):
        self.name_file = name_file
        self.date_frame = pd.read_csv(name_file)

    def reading_column(self, name_column: str) -> list:
        """
        Читает колонку
        :param name_column: наименование колонки
        :return: список значений по колонке
        """
        return self.date_frame[name_column].tolist()
