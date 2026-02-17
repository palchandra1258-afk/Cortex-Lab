"""
Training script for fine-tuning DeepSeek-R1-Distill-Qwen-14B
Supports LoRA, QLoRA, and full fine-tuning
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
OUTPUT_DIR = "./output"
CHECKPOINT_DIR = "./checkpoints"

class ModelTrainer:
    def __init__(
        self,
        model_name=MODEL_NAME,
        use_lora=True,
        use_4bit=False,
        use_8bit=False,
        output_dir=OUTPUT_DIR
    ):
        """
        Initialize the trainer
        
        Args:
            model_name: HuggingFace model name or local path
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            use_4bit: Use 4-bit quantization (QLoRA)
            use_8bit: Use 8-bit quantization
            output_dir: Directory to save outputs
        """
        self.model_name = model_name
        self.use_lora = use_lora
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with appropriate configuration"""
        print("\n" + "="*60)
        print("Loading Model and Tokenizer")
        print("="*60 + "\n")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model loading
        load_kwargs = {
            "trust_remote_code": True,
        }
        
        if self.use_4bit:
            print("Using 4-bit quantization (QLoRA)")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.use_8bit:
            print("Using 8-bit quantization")
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        
        # Load model
        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        
        # Prepare for k-bit training if using quantization
        if self.use_4bit or self.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA if requested
        if self.use_lora:
            self.apply_lora()
        
        print("✓ Model and tokenizer loaded successfully\n")
        
    def apply_lora(self):
        """Apply LoRA configuration to the model"""
        print("Applying LoRA configuration...")
        
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("✓ LoRA applied successfully\n")
        
    def prepare_dataset(self, dataset_name=None, dataset_path=None, text_field="text"):
        """
        Prepare training dataset
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_path: Local dataset path
            text_field: Name of the text field in the dataset
        """
        print("\n" + "="*60)
        print("Preparing Dataset")
        print("="*60 + "\n")
        
        # Load dataset
        if dataset_name:
            print(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)
        elif dataset_path:
            print(f"Loading dataset from: {dataset_path}")
            dataset = load_dataset("json", data_files=dataset_path)
        else:
            # Use a sample dataset for demonstration
            print("No dataset specified. Using sample data...")
            from datasets import Dataset
            sample_data = {
                "text": [
                    "Question: What is 2+2? Answer: Let me think step by step. 2+2 equals 4.",
                    "Question: Explain photosynthesis. Answer: Photosynthesis is the process by which plants convert light energy into chemical energy.",
                ]
            }
            dataset = Dataset.from_dict(sample_data)
            dataset = {"train": dataset}
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_field],
                truncation=True,
                max_length=2048,
                padding="max_length"
            )
        
        print("Tokenizing dataset...")
        self.train_dataset = dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        print(f"✓ Dataset prepared: {len(self.train_dataset)} examples\n")
        
    def train(
        self,
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        warmup_steps=100,
        save_steps=500,
        logging_steps=10,
        use_wandb=False
    ):
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            use_wandb: Use Weights & Biases for logging
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60 + "\n")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            report_to="wandb" if use_wandb else "none",
            logging_dir=f"{self.output_dir}/logs",
            optim="paged_adamw_8bit" if (self.use_4bit or self.use_8bit) else "adamw_torch",
            gradient_checkpointing=True,
            group_by_length=True,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        print("Training started...\n")
        trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model(f"{self.output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final_model")
        
        print("\n" + "="*60)
        print("✓ Training Complete!")
        print("="*60)
        print(f"Model saved to: {self.output_dir}/final_model\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Model name or path")
    parser.add_argument("--use-lora", action="store_true", default=True, help="Use LoRA")
    parser.add_argument("--4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--dataset", type=str, help="Dataset name or path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_name=args.model_name,
        use_lora=args.use_lora,
        use_4bit=args.__dict__['4bit'],
        use_8bit=args.__dict__['8bit']
    )
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Prepare dataset
    trainer.prepare_dataset(dataset_name=args.dataset)
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb
    )

if __name__ == "__main__":
    main()
