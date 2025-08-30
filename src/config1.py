# to make exe file from Qt

from cx_Freeze import setup, Executable

# Specify the script to be converted to an executable
script = 'qttut2.py'

# Define the base options for the executable
base = None

# Create an executable
executables = [Executable(script, base=base)]

# Additional options for the setup
options = {
    'build_exe': {
        'packages': ['qgis','requests','processing'],  # List any packages that your script depends on
        'include_files': ['data'],  # List any additional files or data files
    },
}

# Create the setup
setup(
    name='bcd',
    version='1.0',
    description='Your Description',
    executables=executables,
    options=options
)
