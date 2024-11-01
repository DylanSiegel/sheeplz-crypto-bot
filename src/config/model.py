from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Neural network configuration.

    This class defines the architecture and training hyperparameters for the neural network model.

    Attributes:
        feature_dim (int): The dimensionality of the input feature vector.  This should match the output of your feature extractor.
        hidden_size (int): The number of neurons in each hidden layer of the neural network.
        num_layers (int): The number of hidden layers in the n-LNN (novel neural network) architecture.
        dropout (float): The dropout rate used for regularization during training (a value between 0 and 1).
        learning_rate (float): The learning rate used by the optimizer.  This controls the step size during gradient descent.
        batch_size (int): The size of the mini-batches used during training.  Larger batch sizes can improve training speed but require more memory.
        sequence_length (int): The length of the input sequences used for training the recurrent layers (if applicable).  This is relevant if your model has RNN components.
        gradient_clip (float): The maximum value for gradient clipping to prevent exploding gradients during training.  This helps stabilize training.

    """
    feature_dim: int = Field(64, description="Input feature dimension")
    hidden_size: int = Field(256, description="Hidden layer size")
    num_layers: int = Field(4, description="Number of n-LNN layers")
    dropout: float = Field(0.1, description="Dropout rate")
    learning_rate: float = Field(3e-4, description="Learning rate")
    batch_size: int = Field(512, description="Training batch size")
    sequence_length: int = Field(100, description="Sequence length for training")
    gradient_clip: float = Field(1.0, description="Gradient clipping value")