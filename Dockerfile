FROM satijalab/seurat:latest

# Set environment variables 
# non interactive mode is for dealing with arguements that require response 
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for RStudio Server
RUN apt-get update && apt-get install -y \
    gdebi-core \
    psmisc \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libhdf5-dev \
    libfftw3-dev \
    libgsl-dev \
    libclang-dev \
    wget \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install RStudio Server
RUN wget https://download2.rstudio.org/server/focal/amd64/rstudio-server-2024.12.0-467-amd64.deb && \
    gdebi -n rstudio-server-2024.12.0-467-amd64.deb && \
    rm rstudio-server-2024.12.0-467-amd64.deb

# Expose RStudio Server port
EXPOSE 8787

# Set default user and password
ENV USERNAME=rstudio
ENV PASSWORD=seurat

# Create default user
RUN useradd -m -d /home/$USERNAME -s /bin/bash $USERNAME && \
    echo "$USERNAME:$PASSWORD" | chpasswd && \
    adduser $USERNAME sudo

# Set up volume mounting
VOLUME ["/workspace"]

# Keep the container running
CMD ["/usr/lib/rstudio-server/bin/rserver", "--server-daemonize=0"]
