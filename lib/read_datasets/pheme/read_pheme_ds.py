from lib.read_datasets.pheme.file_dir_handler import FileDirHandler
import lib.constants as constants
from lib.read_datasets.pheme.read_pheme_csv_dataset import ReadPhemeCSVDataset
from lib.read_datasets.pheme.read_pheme_json_dataset import ReadPhemeJsonDataset


def read_pheme_ds():
    print("\n<< PHASE-1 <==> READ DATA >>")

    pheme_csv_dirs = FileDirHandler.read_directories(directory=constants.PHEME_CSV_DIR)

    dataframe = None
    if pheme_csv_dirs is None or not pheme_csv_dirs.__contains__(constants.PHEME_CSV_NAME):
        readPhemeJsonDS = ReadPhemeJsonDataset()
        readPhemeJsonDS.read_and_save_csv()
        dataframe = readPhemeJsonDS.df
    else:
        readPhemeCSVDS = ReadPhemeCSVDataset()
        readPhemeCSVDS.read_csv_dataset()
        dataframe = readPhemeCSVDS.df

    print("\tPath (.csv) : " + constants.PHEME_CSV_PATH)
    print("\tShape (.csv) : " + str(dataframe.shape))
    print("<< PHASE-1 <==> READ DATA DONE>>")

    return dataframe
