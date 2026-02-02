"""
This lets you run RAVES from the command line, using the following syntax:
```
# After pip installation:
$ raves "path/to/your/environment/folder"

# Alternatively (without pip-install, run from root directory):
$ python -m raves "path/to/your/environment/folder"
```
Run it with argument `-h` or `--help` to see a list of optional arguments.
For an example, try
```
raves "./example environments/Shoebox_6_patches"
```
"""
from .cli import main

if __name__ == "__main__":
    main()
