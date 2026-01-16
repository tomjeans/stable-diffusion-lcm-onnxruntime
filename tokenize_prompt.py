#!/usr/bin/env python3
"""
Helper script to tokenize prompts using the real CLIP tokenizer.
This generates token IDs that can be used by the C++ implementation.
"""

import argparse
import numpy as np
from transformers import CLIPTokenizer

def main():
    parser = argparse.ArgumentParser(description="Tokenize a prompt using CLIP tokenizer")
    parser.add_argument("--prompt", required=True, help="Text prompt to tokenize")
    parser.add_argument("--output", required=True, help="Output file path (.npy or .bin)")
    
    args = parser.parse_args()
    
    # Load the CLIP tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    
    # Tokenize the prompt
    text_inputs = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    
    token_ids = text_inputs.input_ids.astype(np.int32)
    
    # Save as numpy array
    if args.output.endswith('.npy'):
        np.save(args.output, token_ids)
    else:
        # Save as binary file
        token_ids.tofile(args.output)
    
    print(f"Tokenized prompt saved to {args.output}")
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"First 10 token IDs: {token_ids[0, :10]}")
    print(f"Last 10 token IDs: {token_ids[0, -10:]}")

if __name__ == "__main__":
    main()
