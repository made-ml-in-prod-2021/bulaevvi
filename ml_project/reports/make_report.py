import pandas as pd
from pandas_profiling import ProfileReport
import logging.config
import yaml
import click
from dataclasses import dataclass
from marshmallow_dataclass import class_schema


APPLICATION_NAME = 'homework01'
logger = logging.getLogger(APPLICATION_NAME)


@dataclass
class ReportParams:
    input_data_path: str
    output_data_path: str
    logger_config: str
   
ReportParamsSchema = class_schema(ReportParams)


@click.command(name="make_report")
@click.argument("config_path")
def make_report(config_path: str):
    # Loading report params
    with open(config_path, "r") as input_stream:
        params = ReportParamsSchema().load(yaml.safe_load(input_stream))
    # Loading logger params
    with open(params.logger_config) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))
    
    logger.info("Start creating data report...")
    input_csv_data = pd.read_csv(params.input_data_path)
    profile = ProfileReport(input_csv_data)
    profile.to_file(output_file = params.output_data_path)
    logger.info("Data report has succesfully created")
    

if __name__ == "__main__":
    make_report()
