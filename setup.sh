#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Create necessary directories and files if they don't exist
touch agent_rewards.txt

# Make main script executable
chmod +x main1.py

echo "Setup complete! To run the simulation:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the simulation: python main1.py"