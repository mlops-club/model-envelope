import os
from pathlib import Path

THIS_DIR = Path(__file__).parent
DATA_DIR = (THIS_DIR / "../../../../data").resolve()
MODELS_DIR = (THIS_DIR / "../../../models").resolve()
MLRUNS_DIR = (THIS_DIR / "../../../mlruns").resolve()

# Set MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = f"file://{MLRUNS_DIR}"

RUNESCAPE_ITEM_PRICES_DATASET_FPATH = DATA_DIR / "Runescape_Item_Prices.csv"
RUNESCAPE_ITEM_NAMES_DATASET_FPATH = DATA_DIR / "Runescape_Item_Names.csv"
RUNESCAPE_ITEM_URLS_DATASET_FPATH = DATA_DIR / "Runescape_Item_URLS.csv"


# Model hyperparameters
WINDOW_SIZE = 5  # Number of days to look back
HIDDEN_SIZE = 64  # Size of hidden layers
NUM_HEADS = 4  # Number of attention heads
NUM_LAYERS = 2  # Number of transformer layers
DROPOUT = 0.2  # Dropout rate

# Training hyperparameters
BATCH_SIZE = 32  # Training batch size
EPOCHS = 100  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate for optimizer

# Other constants
POSITIONAL_ENCODING_MAX_LEN = 1000  # Maximum length for positional encoding
