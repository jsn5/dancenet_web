import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MDNRNN(nn.Module):
    """
    Mixture Density Network RNN for dance pose generation.
    
    This model combines an RNN (LSTM/GRU) with a Mixture Density Network output
    layer to predict probability distributions over possible next poses.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_mixtures, output_size, rnn_type='lstm', dropout=0.0):
        """
        Initialize the MDN-RNN model.
        
        Args:
            input_size (int): Dimension of input features (flattened pose keypoints).
            hidden_size (int): Number of hidden units in the RNN.
            num_layers (int): Number of RNN layers.
            num_mixtures (int): Number of Gaussian mixtures in the MDN.
            output_size (int): Dimension of output features (flattened pose keypoints).
            rnn_type (str): Type of RNN cell ('lstm' or 'gru').
            dropout (float): Dropout probability.
        """
        super(MDNRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.output_size = output_size
        self.rnn_type = rnn_type.lower()
        
        # RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # MDN output layer
        # For each mixture component we need:
        # - 1 value for mixture weight (pi)
        # - output_size values for means (mu)
        # - output_size values for standard deviations (sigma)
        mdn_output_dim = num_mixtures * (1 + 2 * output_size)
        self.mdn_layer = nn.Linear(hidden_size, mdn_output_dim)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the MDN-RNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_size].
            hidden (tuple or torch.Tensor, optional): Initial hidden state.
            
        Returns:
            tuple: (pi, mu, sigma) where
                pi is the mixture weights of shape [batch_size, seq_len, num_mixtures]
                mu is the means of shape [batch_size, seq_len, num_mixtures, output_size]
                sigma is the standard deviations of shape [batch_size, seq_len, num_mixtures, output_size]
                
                Note: If evaluating a single step (seq_len=1), the seq_len dimension will be squeezed.
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Pass input through RNN
        outputs, hidden = self.rnn(x, hidden)
        
        # Pass RNN output through MDN layer
        mdn_outputs = self.mdn_layer(outputs)
        
        # Split MDN outputs into mixture weights, means, and standard deviations
        pi, mu, sigma = self._split_mdn_outputs(mdn_outputs, batch_size, seq_len)
        
        return pi, mu, sigma, hidden
    
    def _split_mdn_outputs(self, mdn_outputs, batch_size, seq_len):
        """
        Split the MDN layer outputs into mixture weights, means, and standard deviations.
        
        Args:
            mdn_outputs (torch.Tensor): MDN layer outputs.
            batch_size (int): Batch size.
            seq_len (int): Sequence length.
            
        Returns:
            tuple: (pi, mu, sigma)
        """
        # Reshape to [batch_size, seq_len, num_mixtures * (1 + 2 * output_size)]
        mdn_outputs = mdn_outputs.view(batch_size, seq_len, -1)
        
        # Split into pi, mu, sigma
        pi_idx = self.num_mixtures
        mu_idx = pi_idx + self.num_mixtures * self.output_size
        
        # Extract mixture weights (pi)
        pi = mdn_outputs[:, :, :pi_idx]
        pi = F.softmax(pi, dim=-1)  # Apply softmax to get valid probabilities
        
        # Extract means (mu)
        mu = mdn_outputs[:, :, pi_idx:mu_idx]
        mu = mu.view(batch_size, seq_len, self.num_mixtures, self.output_size)
        
        # Extract standard deviations (sigma)
        sigma = mdn_outputs[:, :, mu_idx:]
        sigma = sigma.view(batch_size, seq_len, self.num_mixtures, self.output_size)
        sigma = torch.exp(sigma)  # Apply exponential to ensure positive values
        
        return pi, mu, sigma
    
    def sample(self, pi, mu, sigma, temperature=1.0):
        """
        Sample from the predicted mixture distributions.
        
        Args:
            pi (torch.Tensor): Mixture weights of shape [batch_size, num_mixtures].
            mu (torch.Tensor): Means of shape [batch_size, num_mixtures, output_size].
            sigma (torch.Tensor): Standard deviations of shape [batch_size, num_mixtures, output_size].
            temperature (float): Temperature parameter for sampling (higher = more random).
            
        Returns:
            torch.Tensor: Sampled poses of shape [batch_size, output_size].
        """
        batch_size = pi.size(0)
        
        # Apply temperature to mixture weights
        if temperature != 1.0:
            pi = pi / temperature
            pi = F.softmax(pi, dim=-1)
        
        # Sample mixture components based on mixture weights
        # For each batch element, sample a mixture component
        component_indices = torch.multinomial(pi, 1).squeeze()
        
        # Create a mask to index into mu and sigma tensors
        batch_indices = torch.arange(batch_size, device=pi.device)
        
        # Get means and standard deviations for the selected components
        selected_mu = mu[batch_indices, component_indices]
        selected_sigma = sigma[batch_indices, component_indices]
        
        # Scale sigma by temperature (higher temperature = more variance)
        if temperature != 1.0:
            selected_sigma = selected_sigma * temperature
        
        # Sample from Gaussian distributions
        eps = torch.randn_like(selected_mu)
        samples = selected_mu + selected_sigma * eps
        
        return samples
    
    def generate_sequence(self, seed_sequence, sequence_length, temperature=1.0):
        """
        Generate a sequence of poses starting from a seed sequence.
        
        Args:
            seed_sequence (torch.Tensor): Initial seed sequence of shape [1, seed_len, input_size].
            sequence_length (int): Length of sequence to generate.
            temperature (float): Temperature parameter for sampling (higher = more random).
            
        Returns:
            torch.Tensor: Generated sequence of shape [1, seed_len + sequence_length, output_size].
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            # Initialize with the seed sequence
            generated_sequence = seed_sequence.clone()
            current_input = seed_sequence.clone()
            
            # Get the hidden state after processing the seed sequence
            _, hidden = self.rnn(current_input)
            
            # Generate new poses one by one
            for _ in range(sequence_length):
                # Get the last generated pose
                last_pose = current_input[:, -1:, :]
                
                # Feed it through the model
                outputs, hidden = self.rnn(last_pose, hidden)
                mdn_outputs = self.mdn_layer(outputs)
                
                # Split MDN outputs
                pi, mu, sigma = self._split_mdn_outputs(mdn_outputs, 1, 1)
                
                # Squeeze time dimension for sampling
                pi = pi.squeeze(1)
                mu = mu.squeeze(1)
                sigma = sigma.squeeze(1)
                
                # Sample next pose
                next_pose = self.sample(pi, mu, sigma, temperature)
                next_pose = next_pose.unsqueeze(1)  # Add time dimension back
                
                # Append to generated sequence
                generated_sequence = torch.cat([generated_sequence, next_pose], dim=1)
                
                # Update input for next iteration
                current_input = next_pose
        
        return generated_sequence

def mdn_loss_function(pi, mu, sigma, target):
    """
    Calculate the negative log likelihood loss for a mixture density network.
    
    Args:
        pi (torch.Tensor): Mixture weights of shape [batch_size, num_mixtures].
        mu (torch.Tensor): Means of shape [batch_size, num_mixtures, output_size].
        sigma (torch.Tensor): Standard deviations of shape [batch_size, num_mixtures, output_size].
        target (torch.Tensor): Target values of shape [batch_size, output_size].
        
    Returns:
        torch.Tensor: Negative log likelihood loss (scalar).
    """
    batch_size, num_mixtures, output_size = mu.size()
    
    # Expand target to match mu and sigma shapes
    target = target.unsqueeze(1).expand_as(mu)
    
    # Calculate Gaussian probability densities for each mixture component
    # Formula: exponent = -0.5 * ((target - mu) / sigma)^2
    exponent = -0.5 * torch.sum(((target - mu) / (sigma + 1e-6))**2, dim=2)
    
    # Formula: coefficient = 1 / (sigma * sqrt(2π))
    # Log coefficient = -log(sigma) - 0.5 * log(2π)
    log_coefficient = -torch.sum(torch.log(sigma + 1e-6), dim=2) - 0.5 * output_size * np.log(2 * np.pi)
    
    # Combine exponent and coefficient to get log probability density
    log_prob = exponent + log_coefficient
    
    # Weight each component by its mixture weight and sum
    # We use the log-sum-exp trick for numerical stability
    log_pi = torch.log(pi + 1e-6)
    log_likelihood = torch.logsumexp(log_pi + log_prob, dim=1)
    
    # Return negative log likelihood (averaged over batch)
    return -torch.mean(log_likelihood)
