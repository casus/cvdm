import shutil
from pathlib import Path

import numpy as np

# Define source and destination paths
path = "/bigdata/imnet"
paths = np.array(list(Path(path).glob("*.JPEG")))[:100]

# Create destination directory if it doesn't exist
dest_path = Path("data/imnet_sample/")
dest_path.mkdir(parents=True, exist_ok=True)

# Copy each selected image to the destination folder
for img_path in paths:
    shutil.copy(img_path, dest_path / img_path.name)

print(f"Copied {len(paths)} images to {dest_path}")
