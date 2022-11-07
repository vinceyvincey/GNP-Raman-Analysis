# Script for Concatenating Pickle Files

import argparse
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

#  Parser setup
#  --------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="A script to combine pickle df files together."
)

parser.add_argument(
    "-n", "--name", required=False, default="output", help="Name for pickle"
)

args = parser.parse_args()
3 
#  Logging setup
#  --------------------------------------------------------------------------------------------

filename_text = Path.cwd().joinpath("logs", args.name + ".log")

logging.basicConfig(
    filename=filename_text,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

logging.getLogger().setLevel(logging.INFO)

#  Program
#  --------------------------------------------------------------------------------------------

data_dir = Path.cwd().joinpath("data","processed")

def read_files(
    target_directory: Path, 
    file_extension: str, 
    pars: dict = {}
    ):

    """
    Read text files from folders using pandas
    """

    file_list = list(target_directory.rglob("*" + file_extension))
    
    print(file_list)

    df_out = pd.DataFrame()
    for item in tqdm(file_list):

        logging.info(f"processed {item.name}")

        file_in = pd.read_pickle(item, **pars)
        file_in["file"] = item.name
        df_out = pd.concat([df_out, file_in])

    return file_list, df_out


def main(file_dir, pickle_name):
    """
    Concatenate pickle files into single dataframe
    """
    _, df_out = read_files(file_dir, ".pkl")

    pickle_loc = pickle_name + ".pkl"
    df_out.to_pickle(data_dir.joinpath(pickle_loc))

if __name__ == "__main__":

    main(data_dir, args.name)
