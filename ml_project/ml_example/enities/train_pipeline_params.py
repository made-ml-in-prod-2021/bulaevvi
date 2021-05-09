from dataclasses import dataclass
from enities import SplittingParams, FeatureParams, ClassifierParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass
class TrainingPipelineParams:
    input_data_path: str
    input_data_for_validation: str
    output_model_path: str
    output_data_preprocessor_path: str
    predicts_path: str
    logger_config: str
    model_type: str
    split_params: SplittingParams
    feature_params: FeatureParams
    classifier_params: ClassifierParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))