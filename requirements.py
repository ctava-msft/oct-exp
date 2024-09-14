import re

# List of imported modules from the provided code
imported_modules = [
    "os", "shutil", "argparse", "torch", "pytorch_lightning", 
    "ldm", "taming", "datamodule", "torchvision", "utils", "numpy"
]

# Read the requirements.txt file
with open('requirements.txt', 'r') as file:
    requirements = file.readlines()

# Clean up the requirements list
requirements = [re.split('[<>=]', req.strip())[0] for req in requirements]

# Find missing modules
missing_modules = [module for module in imported_modules if module not in requirements]

print("Modules not in requirements.txt:", missing_modules)