# coding: UTF-8
import os
from dotenv import load_dotenv
import pathlib

dotenv_path = pathlib.Path(__file__).parent / '.env'
load_dotenv(dotenv_path)

GCP = os.environ.get("GCP_API_KEY")

if __name__ == "__main__":
    pass
