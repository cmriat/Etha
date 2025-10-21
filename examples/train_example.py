"""Example training script demonstrating Tensor Bus usage."""

from etha.tensor_bus_utils import RPC


class Trainer:
    """Example trainer class for reinforcement learning."""

    def __init__(self):
        """Initialize the trainer."""
        pass

    def forward_backward(self):
        """Execute forward and backward pass."""
        # Placeholder for actual training logic
        pass

    def optimizer_step(self):
        """Execute optimizer step to update weights."""
        # Placeholder for actual optimizer logic
        pass


def main():
    """Main training loop with middleware synchronization."""
    # Connect to middleware
    mw = RPC("http://middleware:8000")
    
    # Initialize trainer
    trainer = Trainer()
    
    step = 0
    max_steps = 100
    
    while step < max_steps:
        # Execute training step
        trainer.forward_backward()
        
        # Wait for middleware to be ready (not busy transferring)
        while mw.get_state() is True:
            pass
        
        # Update optimizer
        trainer.optimizer_step()
        
        # Signal new weights are available
        mw.put()
        
        step += 1


if __name__ == "__main__":
    main()
