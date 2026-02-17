"""
Inference script for DeepSeek-R1-Distill-Qwen-14B
Supports interactive chat and batch inference
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

class DeepSeekInference:
    def __init__(
        self,
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        use_4bit=False,
        use_8bit=False,
        device="auto"
    ):
        """
        Initialize inference engine
        
        Args:
            model_name: Model name or local path
            use_4bit: Use 4-bit quantization
            use_8bit: Use 8-bit quantization
            device: Device to run on
        """
        print("\n" + "="*60)
        print("Initializing DeepSeek-R1-Distill-Qwen-14B")
        print("="*60 + "\n")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Configure model loading
        load_kwargs = {
            "trust_remote_code": True,
        }
        
        if use_4bit:
            print("Using 4-bit quantization")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif use_8bit:
            print("Using 8-bit quantization")
            load_kwargs["load_in_8bit"] = True
        else:
            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = torch.bfloat16
                load_kwargs["device_map"] = device
            else:
                load_kwargs["torch_dtype"] = torch.float32
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        if not use_4bit and not use_8bit:
            self.model.eval()
        
        print("✓ Model loaded successfully\n")
        
    def generate(
        self,
        prompt,
        max_new_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        enforce_think=True
    ):
        """
        Generate response for a prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.5-0.7 recommended)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            enforce_think: Force model to use <think> tags
        """
        # Format prompt with thinking directive if needed
        if enforce_think and "<think>" not in prompt:
            formatted_prompt = f"<think>\n{prompt}"
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from response
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):].strip()
        
        return response
    
    def chat(self):
        """Interactive chat mode"""
        print("\n" + "="*60)
        print("Interactive Chat Mode")
        print("="*60)
        print("\nRecommendations:")
        print("  - Temperature: 0.6 (0.5-0.7 range)")
        print("  - For math problems, ask to put answer in \\boxed{}")
        print("  - Type 'exit' or 'quit' to end\n")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Generate response
                print("\nDeepSeek-R1: ", end="", flush=True)
                response = self.generate(user_input)
                print(response + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n⚠ Error: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-R1-Distill-Qwen-14B Inference")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
                       help="Model name or path")
    parser.add_argument("--4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--prompt", type=str, help="Single prompt for batch inference")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = DeepSeekInference(
        model_name=args.model_name,
        use_4bit=args.__dict__['4bit'],
        use_8bit=args.__dict__['8bit']
    )
    
    # Run inference
    if args.prompt:
        # Batch mode
        print("\nPrompt:", args.prompt)
        print("\nResponse:")
        response = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(response)
    else:
        # Interactive mode
        engine.chat()

if __name__ == "__main__":
    main()
