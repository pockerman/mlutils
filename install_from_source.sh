# remove any previously installed versions
pip uninstall simplemlutils

# build the new package
python3 -m build

# install it
pip install dist/simplemlutils-0.0.1-py3-none-any.whl