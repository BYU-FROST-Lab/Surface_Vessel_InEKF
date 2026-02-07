# load_sbg_types.py

# from __future__ import annotations
from pathlib import Path
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

# --- Configuration ---
# Set the path to the root of your custom message package (sbg_driver in this case).
sbg_driver_path = Path('helpers/sbg_driver')
# IMPORTANT: Replace 'path/to/your/sbg_driver' with the actual path.

# Get a list of all your custom .msg files.
all_sbg_msg_files = list(sbg_driver_path.rglob('*.msg'))

# --- Helper Function ---
# This function guesses the message type string from the file path.
def guess_msgtype(path: Path) -> str:
    """Guess message type name from path."""
    # This logic assumes a standard ROS 2 package structure.
    name = path.relative_to(path.parents[2]).with_suffix('')
    if 'msg' not in name.parts:
        name = name.parent / 'msg' / name.name
    return str(name).replace('\\', '/') # Ensure correct path separator on all OSs

# --- Main Logic ---
# Get a typestore for the desired ROS 2 release.
# Stores.LATEST is a safe choice for recent ROS 2 versions.
typestore = get_typestore(Stores.LATEST)

# Prepare a dictionary to store the definitions to be registered.
custom_types = {}

# print("Found and parsing the following message files:")
for msg_path in all_sbg_msg_files:
    # Read the content of each .msg file.
    msg_def = msg_path.read_text(encoding='utf-8')
    
    # Guess the full message type string.
    msg_type_name = guess_msgtype(msg_path)
    
    # Parse the message definition and add it to the dictionary.
    custom_types.update(get_types_from_msg(msg_def, msg_type_name))
    # print(f"- {msg_path.name} as type '{msg_type_name}'")

# Register the custom types with the typestore.
typestore.register(custom_types)
# print("\nAll custom message types registered successfully! ✅")