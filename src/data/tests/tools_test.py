from src.data.tools import CsvService

def test_csv_read_all_file():
    """
    Проверяет чтение всего файла
    :return:
    """
    csv_service = CsvService("../files/Hyderabad.csv")
    data_columns = csv_service.reading_column("Price")

    assert len(data_columns) != 0