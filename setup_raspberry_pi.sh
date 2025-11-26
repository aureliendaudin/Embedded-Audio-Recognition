#!/bin/bash
# setup_raspberry_pi.sh

echo "Setting up Speech Recognition on Raspberry Pi..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y portaudio19-dev libsndfile1
sudo apt-get install -y libasound2-dev

# Install Python packages
pip3 install -r requirements_pi.txt

# Test microphone
echo "Testing microphone..."
arecord -l

echo "Setup complete!"
echo "Run: python3 raspberry_pi_inference.py"
