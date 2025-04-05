import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.model import MDNRNN, mdn_loss_function
from training.dataset import DancePoseDataset, get_dataloaders

def train_model(model, train_loader, val_loader, optimizer, device, args):
    """
    Train the MDN-RNN model.
    
    Args:
        model (MDNRNN): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): Device to train on.
        args (argparse.Namespace): Training arguments.
        
    Returns:
        tuple: (trained_model, train_losses, val_losses)
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create directory for saving models
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]") as train_pbar:
            for inputs, targets in train_pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                pi, mu, sigma, _ = model(inputs)
                
                # Calculate loss (for the last time step)
                # Reshape from [batch_size, seq_len, ...] to [batch_size, ...]
                pi_last = pi[:, -1, :]
                mu_last = mu[:, -1, :]
                sigma_last = sigma[:, -1, :]
                
                loss = mdn_loss_function(pi_last, mu_last, sigma_last, targets)
                
                # Backward pass and optimize
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
                optimizer.step()
                
                # Update metrics
                batch_loss = loss.item()
                epoch_train_loss += batch_loss
                train_batches += 1
                
                # Update progress bar
                train_pbar.set_postfix({'loss': batch_loss})
        
        avg_train_loss = epoch_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]") as val_pbar:
                for inputs, targets in val_pbar:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Forward pass
                    pi, mu, sigma, _ = model(inputs)
                    
                    # Calculate loss (for the last time step)
                    pi_last = pi[:, -1, :]
                    mu_last = mu[:, -1, :]
                    sigma_last = sigma[:, -1, :]
                    
                    loss = mdn_loss_function(pi_last, mu_last, sigma_last, targets)
                    
                    # Update metrics
                    batch_loss = loss.item()
                    epoch_val_loss += batch_loss
                    val_batches += 1
                    
                    # Update progress bar
                    val_pbar.set_postfix({'loss': batch_loss})
        
        avg_val_loss = epoch_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }
        
        # Save the latest model
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_model.pth'))
        
        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            patience_counter = 0
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, args.save_dir)
    
    # Load the best model
    best_checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    return model, train_losses, val_losses

def plot_loss_curves(train_losses, val_losses, save_dir):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): Training losses for each epoch.
        val_losses (list): Validation losses for each epoch.
        save_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train MDN-RNN model for dance generation")
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='data/processed_poses/dance_dataset_stabilized.npz',
                        help='Path to the preprocessed dataset file')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='Ratio of training data (0.0 to 1.0)')
                        
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Number of hidden units in the RNN')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--num_mixtures', type=int, default=5,
                        help='Number of Gaussian mixtures in the MDN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        help='Type of RNN cell (lstm or gru)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
                        
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--save_dir', type=str, default='training/trained_models',
                        help='Directory to save models')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
                        
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data_path):
        print(f"Error: Dataset file not found at {args.data_path}")
        print("Please run preprocessing first: python training/preprocess.py")
        return
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        num_workers=args.num_workers
    )
    
    # Get input and output sizes from data
    sample_input, sample_target = next(iter(train_loader))
    input_size = sample_input.size(-1)
    output_size = sample_target.size(-1)
    
    print(f"Input size: {input_size}, Output size: {output_size}")
    
    # Initialize model
    model = MDNRNN(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_mixtures=args.num_mixtures,
        output_size=output_size,
        rnn_type=args.rnn_type,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Print model summary
    print(f"\nModel Architecture:\n{model}\n")
    
    # Print training details
    print(f"Training with {len(train_loader.dataset)} samples, validating with {len(val_loader.dataset)} samples")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, Learning rate: {args.lr}")
    
    # Train model
    start_time = time.time()
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, device, args)
    elapsed_time = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")
    print(f"Best model saved to {os.path.join(args.save_dir, 'best_model.pth')}")

if __name__ == "__main__":
    main()
