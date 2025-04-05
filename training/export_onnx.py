import os
import sys
import argparse
import torch
import numpy as np
import onnx
import onnxruntime as ort

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.model import MDNRNN

class MDNRNNWrapper(torch.nn.Module):
    """
    Wrapper class for MDNRNN to make it ONNX exportable.
    This handles the specific requirements for RNN export and MDN calculations.
    """
    
    def __init__(self, mdn_rnn_model):
        """
        Initialize the wrapper with an existing MDN-RNN model.
        
        Args:
            mdn_rnn_model (MDNRNN): The MDN-RNN model to wrap.
        """
        super(MDNRNNWrapper, self).__init__()
        self.model = mdn_rnn_model
        self.model.eval()  # Set to evaluation mode
        
    def forward(self, x, temperature=1.0):
        """
        Forward pass through the wrapped model, returning sampled output.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_size].
            temperature (float): Temperature parameter for sampling.
            
        Returns:
            torch.Tensor: Sampled output of shape [batch_size, output_size].
        """
        # Forward pass through the MDN-RNN model
        pi, mu, sigma, _ = self.model(x)
        
        # Take the last step's predictions
        pi_last = pi[:, -1, :]
        mu_last = mu[:, -1, :]
        sigma_last = sigma[:, -1, :]
        
        # Apply temperature to mixture weights
        if temperature != 1.0:
            pi_last = pi_last / temperature
            pi_last = torch.nn.functional.softmax(pi_last, dim=-1)
        
        # For ONNX export, instead of sampling, we'll:
        # 1. Return all components (pi, mu, sigma) for the last time step
        # 2. JavaScript will handle the sampling
        
        return pi_last, mu_last, sigma_last

def detect_model_params(model_path):
    """
    Detect model parameters from a saved checkpoint.
    
    Args:
        model_path (str): Path to the saved PyTorch model.
        
    Returns:
        dict: Dictionary of model parameters.
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    
    # Extract parameters
    params = {}
    
    # Detect input size from first layer weights
    if 'rnn.weight_ih_l0' in state_dict:
        params['input_size'] = state_dict['rnn.weight_ih_l0'].shape[1]
    
    # Detect hidden size from RNN weights
    if 'rnn.weight_hh_l0' in state_dict:
        params['hidden_size'] = state_dict['rnn.weight_hh_l0'].shape[1]
    
    # Detect number of layers
    num_layers = 1
    while f'rnn.weight_ih_l{num_layers}' in state_dict:
        num_layers += 1
    params['num_layers'] = num_layers
    
    # Detect RNN type (based on weight names/shapes)
    # This is a simplification - for more accuracy check model class name
    params['rnn_type'] = 'lstm'  # Default to LSTM
    
    # Detect output size and num_mixtures from MDN layer
    if 'mdn_layer.weight' in state_dict:
        mdn_output_dim = state_dict['mdn_layer.weight'].shape[0]
        
        # Try to infer output_size and num_mixtures
        # For MDN: mdn_output_dim = num_mixtures * (1 + 2 * output_size)
        if params['input_size'] is not None:
            # Assume output_size = input_size for simplicity
            output_size = params['input_size']
            
            # Calculate num_mixtures
            # mdn_output_dim = num_mixtures * (1 + 2 * output_size)
            # num_mixtures = mdn_output_dim / (1 + 2 * output_size)
            num_mixtures = mdn_output_dim / (1 + 2 * output_size)
            
            # Check if this is a whole number
            if num_mixtures.is_integer():
                params['num_mixtures'] = int(num_mixtures)
                params['output_size'] = output_size
            else:
                # Try other common output_size values
                for output_size in [params['input_size'] // 2, params['input_size'] * 2]:
                    num_mixtures = mdn_output_dim / (1 + 2 * output_size)
                    if num_mixtures.is_integer():
                        params['num_mixtures'] = int(num_mixtures)
                        params['output_size'] = output_size
                        break
                
                # If still not found, use a heuristic
                if 'num_mixtures' not in params:
                    # For MDN with output = input, try this formula
                    params['num_mixtures'] = 3  # Default to 3
                    params['output_size'] = params['input_size']
    
    return params

def export_onnx_model(model_path, output_path, input_size=None, sequence_length=30, hidden_size=None, 
                      num_layers=None, num_mixtures=None, output_size=None, rnn_type=None):
    """
    Export the MDN-RNN model to ONNX format.
    
    Args:
        model_path (str): Path to the saved PyTorch model.
        output_path (str): Path to save the ONNX model.
        input_size (int, optional): Dimension of input features.
        sequence_length (int): Length of input sequences.
        hidden_size (int, optional): Number of hidden units in the RNN.
        num_layers (int, optional): Number of RNN layers.
        num_mixtures (int, optional): Number of Gaussian mixtures in the MDN.
        output_size (int, optional): Dimension of output features.
        rnn_type (str, optional): Type of RNN cell ('lstm' or 'gru').
    """
    # First, try to detect parameters from the model
    detected_params = detect_model_params(model_path)
    
    # Use detected or provided parameters
    input_size = input_size or detected_params.get('input_size')
    hidden_size = hidden_size or detected_params.get('hidden_size')
    num_layers = num_layers or detected_params.get('num_layers')
    num_mixtures = num_mixtures or detected_params.get('num_mixtures')
    output_size = output_size or detected_params.get('output_size')
    rnn_type = rnn_type or detected_params.get('rnn_type')
    
    # Check if we have all required parameters
    required_params = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_mixtures': num_mixtures,
        'output_size': output_size,
        'rnn_type': rnn_type
    }
    
    missing_params = [k for k, v in required_params.items() if v is None]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}. "
                        "Please provide these parameters explicitly.")
    
    print("Model parameters:")
    for param, value in required_params.items():
        print(f"  {param}: {value}")
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Initialize model
    model = MDNRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_mixtures=num_mixtures,
        output_size=output_size,
        rnn_type=rnn_type
    )
    
    # Load state dict from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Create wrapper for ONNX export
    wrapper = MDNRNNWrapper(model)
    
    # Create dummy input (batch_size=1, seq_len=sequence_length)
    dummy_input = torch.randn(1, sequence_length, input_size)
    
    # Define input and output names
    input_names = ["input_sequence"]
    output_names = ["pi", "mu", "sigma"]
    
    # Export to ONNX
    print(f"Exporting model to ONNX format: {output_path}")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input_sequence': {0: 'batch_size', 1: 'sequence_length'},
            'pi': {0: 'batch_size'},
            'mu': {0: 'batch_size'},
            'sigma': {0: 'batch_size'}
        },
        opset_version=12,
        do_constant_folding=True,
        verbose=False
    )
    
    print("ONNX export complete.")
    
    # Verify the exported model
    verify_onnx_model(output_path, dummy_input)

def verify_onnx_model(onnx_path, dummy_input):
    """
    Verify the exported ONNX model by checking its structure and running an inference test.
    
    Args:
        onnx_path (str): Path to the exported ONNX model.
        dummy_input (torch.Tensor): Dummy input for testing.
    """
    # Check that the model is well-formed
    print("Checking ONNX model structure...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure is valid.")
    except Exception as e:
        print(f"Error checking ONNX model: {e}")
        return
    
    # Test inference with ONNX Runtime
    print("Testing ONNX model inference...")
    try:
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Prepare input
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"Input name: {input_name}, shape: {input_shape}")
        
        # Run inference
        dummy_input_numpy = dummy_input.numpy()
        outputs = session.run(None, {input_name: dummy_input_numpy})
        
        # Check outputs
        pi, mu, sigma = outputs
        print(f"Output pi shape: {pi.shape}")
        print(f"Output mu shape: {mu.shape}")
        print(f"Output sigma shape: {sigma.shape}")
        
        # Basic validation checks
        assert pi.shape[1] == mu.shape[1], "Number of mixture components don't match"
        assert mu.shape[2] == sigma.shape[2], "Output feature dimensions don't match"
        
        # Check probability values
        pi_sum = np.sum(pi, axis=1)
        assert np.allclose(pi_sum, 1.0, atol=1e-5), "Mixture weights don't sum to 1"
        
        # Check sigma values (should be positive)
        assert np.all(sigma > 0), "Sigma values should be positive"
        
        print("ONNX model inference test passed.")
    except Exception as e:
        print(f"Error testing ONNX model: {e}")
        return
    
    print(f"ONNX model verification successful. Model saved to: {onnx_path}")

def main():
    parser = argparse.ArgumentParser(description="Export MDN-RNN model to ONNX format")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='training/trained_models/best_model.pth',
                        help='Path to the saved PyTorch model')
    parser.add_argument('--output_path', type=str, default='training/trained_models/dance_rnn.onnx',
                        help='Path to save the ONNX model')
    parser.add_argument('--input_size', type=int, default=None,
                        help='Dimension of input features (will be auto-detected if not provided)')
    parser.add_argument('--output_size', type=int, default=None,
                        help='Dimension of output features (if different from input_size)')
    parser.add_argument('--hidden_size', type=int, default=None,
                        help='Number of hidden units in the RNN')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of RNN layers')
    parser.add_argument('--num_mixtures', type=int, default=None,
                        help='Number of Gaussian mixtures in the MDN')
    parser.add_argument('--rnn_type', type=str, default=None,
                        help='Type of RNN cell (lstm or gru)')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Length of input sequences')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first: python training/train.py")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Export model to ONNX
    export_onnx_model(
        model_path=args.model_path,
        output_path=args.output_path,
        input_size=args.input_size,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_mixtures=args.num_mixtures,
        output_size=args.output_size,
        rnn_type=args.rnn_type
    )

if __name__ == "__main__":
    main()
