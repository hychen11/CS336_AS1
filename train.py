
'''
@Author: hychen11
@Date:   2025-12-03 14:00:00
@Description: 
  • Ability to configure and control the various model and optimizer hyperparameters. done
  • Memory-eﬀicient loading of training and validation large datasets with np.memmap. done
  • Serializing checkpoints to a user-provided path. done
  • Periodically logging training and validation performance (e.g., to console and/or an external service like Weights and Biases). done
'''

import argparse
from ast import parse
import math
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import json
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


from cs336_basics.Transformer import(
    TransformerLm,
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint
)

def main():
  # parse hyperparameters
  args = parse_args()
  
  # set seed
  setup_seed(args.seed)
  
  # set device and dtype
  device = torch.device(args.device)
  dtype_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
  }
  dtype = dtype_map[args.dtype]
  
  # set up checkpoint directory
  os.makedirs(args.checkpoint_path, exist_ok=True)
  
  # set up experiment logging
  # TODO: implement experiment logging setup

  # load data
  print(f"Loading training data from {args.data_path}...")
  training_data = load_data(args.data_path)
  print(f"Loading training data finished, size: {len(training_data):,} tokens.")
  
  print(f"Loading validation data from {args.val_data_path}...")
  validation_data = load_data(args.val_data_path)
  print(f"Loading validation data finished, size: {len(validation_data):,} tokens.")
  
  max_training_tokens = int(np.max(training_data)) if len(training_data)>0 else 0
  max_validation_tokens = int(np.max(validation_data)) if len(validation_data)>0 else 0
  max_token_id = max(max_training_tokens, max_validation_tokens)
  print(f"Maximum token ID in data: {max_token_id}")

  if args.vocab_size <= max_token_id:
    print(f"Vocab size {args.vocab_size} is not large enough to cover max token ID {max_token_id} in the dataset.")
    print(f"Change vocab size into {max_token_id}+1.")
    args.vocab_size = max_token_id + 1
    
  # initialize model
  print("\nInitializing model...")
  model = TransformerLm(
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=args.rope_theta,
    device=device,
    dtype=dtype
  )
  
  num_params = get_parameters_count(model)
  print(f"Model initialized with {num_params:,} parameters.")

  # initialize optimizer
  optimizer = optim.AdamW(
    model.parameters(),
    lr=args.max_learning_rate,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay
  )
  
  start_iter = 0
  # load from checkpoint if provided
  if args.checkpoint_load_path is not None:
    print(f"Loading checkpoint from {args.checkpoint_load_path}...")
    start_iter = load_checkpoint(args.checkpoint_load_path, model, optimizer)
    print(f"Starting training from iteration {start_iter}.")
    
  # Training loop
  print("==========Starting training...==========")
  model.train()
  
  hyperparam_path = os.path.join(args.checkpoint_path, 'hyperparameters.json')
  with open(hyperparam_path, 'w') as f:
    json.dump(vars(args), f, indent=2)
  
  start_time = time.time()
  
  for iter_num in range(start_iter, args.max_iters):
    # get lr and update lr
    lr = get_lr_cosine_schedule(
      iter_num,
      args.max_learning_rate,
      args.min_learning_rate,
      args.warmup_iters,
      args.max_iters
    )
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    
    input, output = get_batch(training_data, args.batch_size, args.context_length, device)
    
    # forward pass, input and output shape is (batch_size, context_length), logits shape is (batch_size, context_length, vocab_size)
    logits = model(input)
    
    # compute cross entropy loss, logits.size(-1) is the vocab size, the last dimension, so logits final shape is (batch_size*context_length, vocab_size)
    logits = logits.view(-1, logits.size(-1))
    # output shape is (batch_size*context_length), since view(-1) will flatten the tensor
    output = output.view(-1)
    loss = cross_entropy(logits, output)
    
    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # gradient clipping
    gradient_clipping(model.parameters(), args.grad_clip)
    
    # optimizer step
    optimizer.step()
    
    # step1: validation
    if iter_num % args.eval_interval == 0:
      val_loss = compute_validation_loss(model, validation_data, args.eval_iters, args.batch_size, args.context_length, device)
      val_perplexity = math.exp(val_loss)
      print(f"Interation {iter_num}: validation loss = {val_loss:.4f}, perplexity = {val_perplexity:.4f}")
    
    # step2: checkpointing
    if iter_num > 0 and iter_num % args.checkpoint_interval == 0:
      checkpoint_path = os.path.join(args.checkpoint_path, f"checkpoint_iter_{iter_num}.pth")
      save_checkpoint(model, optimizer, iter_num, checkpoint_path)
      print(f"Saved checkpoint at iteration {iter_num} to {checkpoint_path}") 
         
    # step3: logging
    if iter_num % args.log_interval == 0:    
      elapsed = time.time() - start_time
      iters_per_sec = (iter_num - start_iter + 1) / elapsed
      
      # total norm is used to check gradient
      """ 
      # use cpu to compute norm, not so good
      total_norm = 0.0
      for p in model.parameters():
        if p.grad is not None:
          param_norm = p.grad.data.norm(2)
          total_norm += param_norm.item() ** 2
      total_norm = total_norm ** 0.5
      """
      # use gpu to compute norm, torch default is L2 norm,  torch.norm is L2 norm
      total_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
      print(f"Iteration {iter_num}: training loss = {loss.item():.4f}, lr = {lr:.6e}, time per iter = {iters_per_sec:.4f}s, grad norm = {total_norm:.4f}")
    
    # TODO: add experiment tracking
    
   # final validation
  final_loss = compute_validation_loss(model, validation_data, args.eval_iters, args.batch_size, args.context_length, device)
  final_perplexity = math.exp(final_loss)
  print(f"Final validation loss = {final_loss:.4f}, Final perplexity = {final_perplexity:.4f}")
      
  # final checkpoint
  final_path = os.path.join(args.checkpoint_path, f"checkpoint_final.pth")
  save_checkpoint(model, optimizer, args.max_iters, final_path)
  print(f"Training completed. Final checkpoint saved to {final_path}")
  
  
    
    
def compute_validation_loss(model: nn.Module, validation_data: np.memmap, eval_iters: int, batch_size: int, context_length: int, device: torch.device) -> float:
  model.eval()
  losses = []
  with torch.no_grad():
    for _ in range(eval_iters):
      val_input, val_output = get_batch(validation_data, batch_size, context_length, device)
      val_logits = model(val_input)
      val_logits = val_logits.view(-1, val_logits.size(-1))
      val_output = val_output.view(-1)
      
      val_losses = cross_entropy(val_logits, val_output)
      losses.append(val_losses.item())
  model.train()
  return np.mean(losses)

def get_parameters_count(model: nn.Module) -> int:
  return sum(p.numel() for p in model.parameters())  

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  
def load_data(path: str, dtype=None) -> np.memmap:  
  # get_batch(dataset: torch.Tensor, batch_size: int, context_length: int, device: torch.device) 
  if path.endswith('.npy'):
    mmap_data = np.load(path, mmap_mode='r')
    print(f"\t load .npy file with shape {mmap_data.shape} and dtype {mmap_data.dtype}")
    return mmap_data
  if dtype is None:
    dtype = np.int32
  # read binary file
  
  num_tokens = os.path.getsize(path) // np.dtype(dtype).itemsize
  mmap_data = np.memmap(path, dtype=dtype, mode='r',shape=(num_tokens,))
  return mmap_data

def parse_args():
  parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
  # model hyperparameters
  parser.add_argument("--vocab_size", type=int, default=10000, help="Size of vocabulary")
  parser.add_argument("--context_length", type=int, default=128, help="Maximum context length")
  parser.add_argument("--d_model", type=int, default=768, help="Dimension of model embeddings")
  parser.add_argument("--num_layers", type=int, default=12, help="Number of Transformer layers")
  parser.add_argument("--num_heads", type=int, default=12, help="Number of Attention heads")
  parser.add_argument("--d_ff", type=int, default=3072, help="Dimension of Feedforward network, default is 4*d_model")
  parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

  # training hyperparameters
  parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
  parser.add_argument("--max_iters", type=int, default=100000, help="Maximum number of training iterations")
  parser.add_argument("--max_learning_rate", type=float, default=3e-4, help="Maximum learning rate")
  parser.add_argument("--min_learning_rate", type=float, default=1e-5, help="Minimum learning rate")
  parser.add_argument("--warmup_iters", type=int, default=1000, help="Number of warmup iterations for learning rate schedule")
  # AdamW hyperparameters
  parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
  parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for AdamW optimizer")
  parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for AdamW optimizer")
  parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
  
  # dataset  
  parser.add_argument("--data_path", type=str, required=True, help="Path to the training data (np.memmap format)")
  parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data (np.memmap format)")
  
  # checkpointing
  parser.add_argument("--checkpoint_path", type=str, default="./checkpoints", help="Directory to save checkpoints")
  parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Interval (in iterations) to save checkpoints")
  parser.add_argument("--checkpoint_load_path", type=str, default=None, help="Path to a checkpoint to load")
  
  # logging
  parser.add_argument('--log_interval', type=int, default=100, help='Log training metrics every N iterations')
  parser.add_argument('--eval_interval', type=int, default=500, help='Evaluate on validation set every N iterations')
  parser.add_argument('--eval_iters', type=int, default=100, help='Number of iterations for validation evaluation')
  
   # Device and precision
  parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on (cuda/cpu)')
  parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'], help='Data type for model parameters')
  
  # experiment tracking
  parser.add_argument('--experiment_name', type=str, default=None, help='Name for the experiment (defaults to timestamp)')
  parser.add_argument('--experiment_log_dir', type=str, default="experiment_logs", help='Directory to save experiment logs')
  parser.add_argument('--wandb_project', type=str, default=None, help='Weights & Biases project name (optional)')
  parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name (optional)')
  parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging even if installed')
  
  # random seed
  parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
  
  args = parser.parse_args()
  return args

if __name__ == '__main__':
    main()
  
