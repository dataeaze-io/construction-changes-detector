#!/bin sh
set -vx

source .venv/bin/activate
uv sync

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129


sudo apt install gnupg software-properties-common
sudo mkdir -m755 -p /etc/apt/keyrings  # not needed since apt version 2.4.0 like Debian 12 and Ubuntu 22 or newer
sudo wget -O /etc/apt/keyrings/qgis-archive-keyring.gpg https://download.qgis.org/downloads/qgis-archive-keyring.gpg

cat << EOF | sudo tee /etc/apt/sources.list.d/qgis.sources
Types: deb deb-src
URIs: https://qgis.org/debian
Suites: your-distributions-codename
Architectures: amd64
Components: main
Signed-By: /etc/apt/keyrings/qgis-archive-keyring.gpg
EOF

sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt update

sudo apt update
sudo apt install qgis qgis-plugin-grass
sudo apt install qgis-server --no-install-recommends --no-install-suggests
apt install python3-qgis

source .venv/bin/activate
