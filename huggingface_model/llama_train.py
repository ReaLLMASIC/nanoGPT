import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import llama_model

# 4. Prepare a dataset for language modeling.
# Here we use wikitext-2 for demonstration. In practice, use your target dataset.
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Add a new pad token
llama_model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
llama_model.model.resize_token_embeddings(len(llama_model.tokenizer))

print(llama_model.model.config)

# Concatenate the text of the train split.
def preprocess_function(examples):
    # Concatenate all texts.
    return llama_model.tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# Use a data collator that will dynamically pad the inputs.
data_collator = DataCollatorForLanguageModeling(tokenizer=llama_model.tokenizer, mlm=False)

# 5. Set up the training arguments.
training_args = TrainingArguments(
    output_dir="./llama_custom_finetune",
    overwrite_output_dir=True,
    num_train_epochs=1,           # Adjust the number of epochs.
    per_device_train_batch_size=4,  # Adjust based on your GPU memory.
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=True
)

# 6. Initialize the Trainer.
trainer = Trainer(
    model=llama_model.model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# 7. Fine-tune the model.
trainer.train()

# 8. Save the finetuned model.
llama_model.model.save_pretrained("llama_custom_finetuned")
llama_model.tokenizer.save_pretrained("llama_custom_finetuned")