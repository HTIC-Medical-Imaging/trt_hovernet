apt-get -y update && apt-get -y upgrade && apt-get install -y build-essential
apt install -y libopenjp2-7 libopenjp2-tools
pip install -r /setup_base/requirements.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y openslide-tools	
apt-get -y install python3-openslide
