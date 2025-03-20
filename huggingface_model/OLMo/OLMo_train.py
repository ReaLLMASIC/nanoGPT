import argparse
from transformers import AutoTokenizer, OlmoForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from OLMo_model import apply_custom_attention

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Olmo with custom attention.")
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=["clipped", "relu", "relu_norm", "gated", "softplus", "softplus_norm", "sigmoid", "sigmoid_norm", "obo", "learned_obo"],
        default="relu",
        help="Select which attention variant to use."
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Select learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Select weight decay.")
    parser.add_argument("--max_steps", type=int, default=20000, help="Total number of training steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Number of steps before an evaluation is done on the eval dataset")
    parser.add_argument("--save_steps", type=int, default=10000, help="Number of steps before a checkpoint is created")
    args = parser.parse_args()

    # Load the pretrained model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
    model = OlmoForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf")

    # Apply the custom attention modifications based on the chosen variant.
    if args.model_variant == "clipped":
        # You can set specific parameters for clipped variant.
        model = apply_custom_attention(model, attention_variant="clipped", zeta=1.0, gamma=-0.03)
    else:
        model = apply_custom_attention(model, attention_variant=args.model_variant)

    # (Optional) Example usage: tokenize a sample input and run a forward pass.
    sample_text = "Hello, how are you today?"
    inputs = tokenizer(sample_text, return_tensors="pt")
    outputs = model(**inputs)
    print("Logits shape:", outputs.logits.shape)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Add a new pad token if the tokenizer doesn't have one
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Tokenize the dataset and add labels for language modeling
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )
        # Replace pad tokens with -100 so they are ignored in loss computation.
        tokenized_inputs["labels"] = [
            [token if token != tokenizer.pad_token_id else -100 for token in input_ids]
            for input_ids in tokenized_inputs["input_ids"]
        ]
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(eval_dataset))

    def filter_valid_examples(example):
        # Count valid tokens (i.e., tokens not equal to -100)
        valid_token_count = sum(1 for token in example["labels"] if token != -100)
        return valid_token_count > 0

    # Filter the evaluation dataset
    filtered_eval_dataset = eval_dataset.filter(filter_valid_examples)
    print("Filtered eval dataset size:", len(filtered_eval_dataset))

    # Create a data collator with dynamic padding (reduces wasted memory on padded tokens)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments with a reduced batch size and gradient accumulation
    training_args = TrainingArguments(
        output_dir="./olmo-" + args.model_variant,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        # num_train_epochs=3,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_dir="./logs",
        logging_steps=100,
        fp16=True,
    )

    # Initialize the Trainer with the data collator for dynamic padding
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=filtered_eval_dataset,
        data_collator=data_collator,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./fine-tuned-olmo-" + args.model_variant)
    tokenizer.save_pretrained("./fine-tuned-olmo-" + args.model_variant)

if __name__ == "__main__":
    main()