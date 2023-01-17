from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, pipeline
from datasets import load_dataset, Dataset

import numpy as np
import evaluate
import logging

log = logging.getLogger('expert_system_gpt_pipeline')


def get_base_model(base_model_name: str):
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto", device_map="auto")
    model.config.pad_token_id = model.config.eos_token_id
    return model


def get_tokenizer(base_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_text_data(tokenizer: AutoTokenizer, file_path: str):
    log.info("Loading dataset")
    current_dataset = load_dataset("text", data_dir="data/01_raw", sample_by="paragraph")

    current_dataset['train'] = current_dataset['train']

    def tokenize_function(examples):
        current_tokenizer_result = tokenizer(examples["text"], padding="max_length", truncation=True)
        return current_tokenizer_result

    log.info("Splitting and tokenizing dataset")
    tokenized_datasets = current_dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"]  # .select(range(1)) # Comment this back out to run all data in pipeline
    return small_train_dataset


def train_expert_system_gpt(tokenizer: AutoTokenizer, small_train_dataset: Dataset, model: AutoModelForCausalLM):
    log.info("Preparing training arguments")

    training_args = TrainingArguments(output_dir="data/07_model_output",
                                      report_to='all',
                                      logging_dir='./logs',
                                      per_device_train_batch_size=1,
                                      label_names=['input_ids', 'attention_mask'],  # 'logits', 'past_key_values'
                                      num_train_epochs=1,
                                      no_cuda=True
                                      )

    def compute_metrics(eval_preds):
        metric = evaluate.load("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_train_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    log.info("Starting training")
    results = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained("data/07_model_output")
    return results.metrics, model
