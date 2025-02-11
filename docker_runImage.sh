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
#   5. Build the Docker image from the Dockerfile if it does not already exist.
#   6. Run the Docker container from the built image.
#   7. Verify that the Docker container is running and accessible at http://localhost:8787.
#   8. Wait for user input to exit and remove the Docker container.
#   9. Remove the running Docker container.
#
# Requirements:
#   - Docker installed and running on the system.
#   - The Dockerfile and supporting files in the same directory as the script.
#
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
  # Install Docker
  sudo apt-get install -yq docker


  # Check if Docker installation was successful
  if ! command -v docker &> /dev/null; then
    echo "Failed to install Docker. Exiting."
    exit 1
  fi
fi


# Check if the Docker image already exists
if sudo docker images ls -q "$IMAGE_NAME" &> /dev/null; then
  echo "The $IMAGE_NAME image already exists. Skipping build."
else
  # Build the Docker image
  echo "Building the Docker image..."
  sudo docker pull "$IMAGE_NAME"
fi


# Run the Docker container
echo "Running the Docker container..."
sudo docker run -d --rm --user root -p "$C_PORT:$C_PORT" -v "$(pwd):/home/rstudio/data_dir" -e "PASSWORD=$RSTUDIO_PASSWORD" "$IMAGE_NAME"



# Check if the Docker container is running
if sudo docker ps | grep -q "$IMAGE_NAME"; then
    echo "
    Docker container running at http://localhost:$C_PORT
    "
else
  echo "Docker container not running"
  exit 1
fi


# Keep the program running until the user wants to exit
echo "Press Enter to exit and remove the Docker container"
read -rp ""

# Remove the running Docker container
echo "Removing the Docker container..."
CONTAINER_ID=$(sudo docker ps -q --filter ancestor="$IMAGE_NAME")
if [ -n "$CONTAINER_ID" ]; then
  sudo docker rm -f "$   CONTAINER_ID"
  echo "Docker container removed successfully"
else
  echo "No Docker container found to remove"
fi
