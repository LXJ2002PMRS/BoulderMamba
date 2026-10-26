"""Starting settings for the public training interface.

These are editable training defaults, not a claimed reconstruction of every
historical optimization stage. Supply data and checkpoints explicitly.
"""
EPOCHS = 12
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
ACCUMULATION_STEPS = 4
GRADIENT_CLIP = 5.0
SEED = 42
