"""
This is a boilerplate pipeline 'train_expert_system'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_tokenizer, get_base_model, prepare_text_data, train_expert_system_gpt


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_base_model,
            inputs="params:base_model_name",
            outputs="model",
            name="get_base_model",
        ),
        node(
            func=get_tokenizer,
            inputs="params:base_model_name",
            outputs="tokenizer",
            name="get_tokenizer",
        ),
        node(
            func=prepare_text_data,
            inputs=["tokenizer", "expert_system_training_text"],
            outputs="small_train_dataset",
            name="prepare_text_data",
        ),
        node(
            func=train_expert_system_gpt,
            inputs=["tokenizer", "small_train_dataset", "model"],
            outputs=["expert_system_gpt_pipeline.train_expert_system.metrics", "model_trained"],
            name="train_expert_system_gpt",
        ),
    ])
