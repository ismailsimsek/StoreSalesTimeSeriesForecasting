import logging
import shutil
import zipfile
from pathlib import Path

import kaggle


class Utils:
    @staticmethod
    def download(datadir: Path, competition: str, clean_first: bool = False) -> None:
        if clean_first:
            logging.info("Removing data directory: %s" % datadir.as_posix())
            shutil.rmtree(datadir.as_posix())
        if not datadir.joinpath("train.csv").exists():
            kaggle.api.authenticate()
            kaggle.api.competition_download_files(competition=competition, path=datadir.as_posix())
            with zipfile.ZipFile(datadir.joinpath("%s.zip" + competition).as_posix(), 'r') as zip_ref:
                zip_ref.extractall(datadir.as_posix())
