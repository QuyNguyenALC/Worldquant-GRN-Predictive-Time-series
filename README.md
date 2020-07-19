Detail step to setup and run this project:
1. Clone this project with command line:
git clone https://github.com/QuyNguyenALC/Worldquant-GRN-Predictive-Time-series.git
2. Make sure docker is installed in your computer
3. Build image docker with:
docker build -t docker-model -f Dockerfile .
4. Run the reference.py to predict:
docker run docker-model python3 inference.py
