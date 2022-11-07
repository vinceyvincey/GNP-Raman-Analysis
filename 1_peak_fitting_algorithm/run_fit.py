"""
Peak Fitting Script
"""
import argparse
import logging
from pathlib import Path
from modules.high_level_fitting import pre_processing, curve_fitting, export_curve_fittings

#  Parser setup
#  --------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="A script to fit Raman peaks.")

parser.add_argument(
    "-f",
    "--file_path",
    required=True,
    help="Path to the .wdf file",
)

parser.add_argument(
    "-p", "--pass_mark", required=False, default=0.975, help="Significance cut-off"
)

parser.add_argument(
    "-l1", "--level_1", required=False, default="Material", help="Material"
)

parser.add_argument(
    "-l2", "--level_2", required=False, default="Functionalisation", help="Functionalisation"
)

parser.add_argument(
    "-l3", "--level_3", required=False, default="Date", help="Plasma Run Date"
)

parser.add_argument(
    "-n", "--name", required=False, default="Output", help="Name for Output Pickle File"
)

args = parser.parse_args()

#  Logging setup
#  --------------------------------------------------------------------------------------------

filename_text = 'logs/' + args.name + ".log"

logging.basicConfig(
    filename=filename_text,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

logging.getLogger().setLevel(logging.INFO)

#  Program
#  --------------------------------------------------------------------------------------------


def main(file_path, pass_mark, level_1, level_2, level_3, out_name):

    """
    Peak Fitting Script
    """
    
    logging.info(f'Processing {out_name}')
    
    #  Pre processing
    #  --------------------------------------------------------------------------------------------

    df_ready, df_errors, generic_x = pre_processing(file_path,700,3500)
    
    # df_ready = df_ready.iloc[0:20]
    
    logging.info("data preprocessed")

    #  Curve Fitting
    #  --------------------------------------------------------------------------------------------

    df_final_fits = curve_fitting(
        df_ready, 
        df_errors, 
        pass_mark, 
        generic_x, 
        Path.cwd().joinpath('constraints','gnp_constraints.yaml'),
        )
    
    logging.info("fitting round 3 complete")

    #  Export
    #  --------------------------------------------------------------------------------------------
    export_curve_fittings(df_final_fits, level_1, level_2, level_3, out_name)
    
    logging.info("program complete")


if __name__ == "__main__":

    main(
        args.file_path,
        args.pass_mark,
        args.level_1,
        args.level_2,
        args.level_3,
        args.name,
    )
