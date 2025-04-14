import argparse, torch
from transformers import AutoTokenizer, OlmoForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from OLMo_model import apply_custom_attention
from OLMo_train_args import parse_args
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert

def main():
    args = parse_args()

    # Load the pretrained model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
    model = OlmoForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf")

    # Apply the custom attention modifications based on the chosen variant.
    if args.clip:
        # You can set specific parameters for clipped variant.
        model = apply_custom_attention(model, clip=True, gate=args.gate, obo_variant=args.obo_variant, attention_variant=args.model_variant, zeta=1.0, gamma=-0.03)
    else:
        model = apply_custom_attention(model, clip=False, gate=args.gate, obo_variant=args.obo_variant, attention_variant=args.model_variant)

    # (Optional) Example usage: tokenize a sample input and run a forward pass.
    sample_text = "Hello, how are you today?"
    inputs = tokenizer(sample_text, return_tensors="pt")
    outputs = model(**inputs)
    print("Logits shape:", outputs.logits.shape)

    if args.quantize:
        # Put the model in training mode.
        model.train()
        # Set the default QAT configuration. "fbgemm" is a common backend for CPU QAT.
        model.qconfig = get_default_qat_qconfig("fbgemm")
        # Disable quantization on embedding layers.
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                module.qconfig = None
        model = model.float()
        # Prepare the model for QAT (this inserts fake quantization modules).
        prepare_qat(model, inplace=True)
        # (Optional) Move the model to CPU for QAT training.
        model.to("cpu")

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

    model_variant = (
        ("clipped_" if args.clip else "") +
        ("gated_" if args.gate else "") +
        (args.obo_variant + "_" if args.obo_variant != "none" else "") +
        args.model_variant
    )
    output_dir = "olmo-" + model_variant
    if args.quantize:
        output_dir = "quant_" + output_dir
    fp16_flag = False if args.quantize else True
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        # num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        max_steps=10000,
        save_steps=10000,
        logging_dir="./logs",
        logging_steps=100,
        fp16=fp16_flag,
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

    if args.quantize:
        # After training, convert the model to a quantized version.
        model = convert(model.eval(), inplace=False)

    # Save the fine-tuned model and tokenizer
    if args.quantize:
        save_dir = "./fine-tuned-quant-olmo-" + model_variant
    else:
        save_dir = "./fine-tuned-olmo-" + model_variant
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()