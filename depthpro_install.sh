# Install DepthPro and dependencies
pip install -r requirements_dp.txt
pip install -e ./DepthPro/

mkdir -p checkpoints
curl https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -o checkpoints/depth_pro.pt
