import sys

def normalize(x, x_min, x_max):
    """Normalize x to the range [-1, 1]."""
    return 2 * (x - x_min) / (x_max - x_min) - 1

def inverse_normalize(x_norm, x_min, x_max):
    """Inverse normalization: convert x_norm from [-1, 1] back to the original range."""
    return ((x_norm + 1) / 2) * (x_max - x_min) + x_min

def map_to_uint8(x_norm):
    """Map values from [-1, 1] to [0, 255] for grayscale image saving."""
    return ((x_norm + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

def inverse_map_from_uint8(img_uint8):
    """
    Convert uint8 image data ([0, 255]) back to the normalized range [-1, 1].
    """
    return (img_uint8.astype(np.float32) / 255) * 2 - 1

def map_to_uint10(x_norm):
    """Map normalized data from [-1, 1] to 10-bit range [0, 1023]."""
    return np.clip(((x_norm + 1) / 2 * 1023), 0, 1023).astype(np.uint16)

def inverse_map_from_uint10(img_uint10):
    """
    Convert uint10 image data ([0, 1023]) back to the normalized range [-1, 1].
    """
    return (img_uint10.astype(np.float32) / 1023) * 2 - 1

class Logger(object):
    """
    Write prints in a log file
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message) 
        self.log.write(message) 

    def flush(self):
        self.terminal.flush()
        self.log.flush()


