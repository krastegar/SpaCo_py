#!/bin/bash


# =============================================================================
# Seurat RStudio Server Docker Deployment Script
# =============================================================================
#
# This script builds and deploys a Seurat RStudio Server Docker container.
#
# Algorithm:
#   1. Check if Docker is installed and running on the system.
#   2. Install Docker if it is not already installed.
#   3. Check if the Dockerfile exists in the current directory.
#   4. Check if the Docker image already exists.
#   5. Pull the Docker image from the Docker Hub if it does not already exist.
#   6. Run the Docker container.
#   7. Verify that the Docker container is running and accessible at http://localhost:8787.
#   8. Wait for user input to exit and remove the Docker container.
#   9. Remove the running Docker container.
#
###############################################################################
# Author: Kiarash Rastegar
# GitHub: https://github.com/krastegar
# Date: 01.02.2025
# =============================================================================


# Set the name of the Dockerfile to use
#DOCKERFILE_NAME="Dockerfile"


# Set the name of the Docker image to build
IMAGE_NAME="krastegar0/seurat_rstudio-server:1.0.2"


# Set the port to use for the Docker container
C_PORT=8787


# Set the password for the RStudio server
# Load variables from .env file
if [ -f .env ]; then
    export "$(grep -v '^#' .env | xargs)"
fi


# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  echo "Docker is not installed. Installing Docker..."

  # Install Latest stable version of docker on Ubuntu
  sudo apt update
  sudo apt install -y ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc > /dev/null
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt update
  sudo apt install -y docker-ce docker-ce-cli containerd.io

else
    echo "Docker is already installed."
fi

# Step 2: Ensure Docker is running
if (! systemctl is-active --quiet docker); then
    echo "Starting Docker service..."
    sudo systemctl start docker
fi

# Check if the Docker image already exists
if sudo docker images ls -q "$IMAGE_NAME" &> /dev/null; then
  echo "The $IMAGE_NAME image already exists. Skipping build / pull."
else
  # Build the Docker image
  echo " Pulling the Docker image..."
  sudo docker pull "$IMAGE_NAME"
fi

# Run the Docker container
# Extract the latest image ID for the specified image
IMAGE_ID=$(docker images -q "$IMAGE_NAME" | head -n 1)

# Generate a dynamic container name based on the image ID
CONTAINER_NAME="container_${IMAGE_ID:0:12}"  # Using the first 12 characters of the image ID
echo "Running the Docker container..."
sudo docker run -d --rm --user root -p "$C_PORT:$C_PORT" -v "$(pwd):/home/rstudio/data_dir" -e "PASSWORD=$RSTUDIO_PASSWORD" --name "$CONTAINER_NAME" "$IMAGE_NAME"

# Check if the Docker container is running
if sudo docker ps | grep -q "$IMAGE_NAME"; then
    echo "
    Docker container running at http://localhost:$C_PORT
    "
else
  echo "Docker container not running"
  exit 1
fi

# Wait for user input to exit and remove the container
echo "Press any key to stop and remove the container..."
read -n 1 -s

# stop the container to remove it 
echo "Stopping and removing the container..."
docker stop "$CONTAINER_NAME"

# Finish clean up of the container
echo "Cleanup complete. Exiting."