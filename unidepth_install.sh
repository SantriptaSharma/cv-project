# Install UniDepth and dependencies
pip install -e ./UniDepth/ --extra-index-url https://download.pytorch.org/whl/cu118

# Install Pillow-SIMD (Optional)
# pip uninstall pillow
# CC="cc -mavx2" pip install -U --force-reinstall pillow-simd