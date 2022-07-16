import requests
from tqdm import tqdm
import hashlib
import os
import zipfile
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:[%(levelname)s]:%(name)s:%(message)s')

file_handler = logging.FileHandler('data.log', 'w')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

class Downloader():
    def __init__(self, url, checksum, file, path=""):
        self.url = url
        self.checksum = checksum
        self.file = file
        self.path = path

    def _get(self):
        logger.debug(f"downloading {self.url}")
        res = requests.get(self.url, stream=True)
        total = int(res.headers.get('content-length', 0))

        with open(self.file, 'wb') as file, tqdm(desc=self.file, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for data in res.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    def _checksum(self, blocksize=2**20):
        logger.debug(f"digeting checksum")
        with open(self.file, 'rb') as file:
            hash = hashlib.md5()
            while True:
                byte = file.read(blocksize)
                if not byte:
                    break
                hash.update(byte)
        logger.debug(f"digest {hash.hexdigest()}")
        return hash.hexdigest()

    def _verify(self):
        logger.debug(f"verifying checksum")
        return self._checksum() != self.checksum

    def _extract(self):
        logger.debug(f"extracting file")
        with zipfile.ZipFile(self.file, "r") as archive:
            archive.extractall(self.path)
        
    def fetch(self):
        if os.path.exists(self.path):
            logger.debug(f"{self.path} already exist!")
            return
        if os.path.exists(self.file):
            logger.debug(f"{self.file} already exist!")
            if self._verify():
                logger.warn(f"corrupted file!")
                self._get()
        else:
            self._get()
        if self._verify():
            logger.exception("failed to verify")
        else:
            if not zipfile.is_zipfile(self.file):
                logger.exception("not a zip file")
            else:
                self._extract()
                logger.debug("extract successful")
                os.remove(self.file)
                logger.debug("file removed")
        