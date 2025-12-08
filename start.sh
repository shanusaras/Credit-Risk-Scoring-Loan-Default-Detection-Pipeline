#!/bin/bash
set -e  # exit immediately if any command fails

# Create geoip_db directory (if it doesn't exist already)
mkdir -p /app/geoip_db

# Download GeoLite2-Country.mmdb (if it doesn't exist already)
if [ ! -f /app/geoip_db/GeoLite2-Country.mmdb ]; then
  echo "Downloading GeoLite2-Country.mmdb..."
  
  # Create an account at https://www.maxmind.com/, create a license key and add it to your .env file and your Hugging Face Space secrets
  # Sanitize the key: remove single/double quotes and whitespace (Docker --env-file includes quotes in the value, which breaks the URL)
  MAXMIND_LICENSE_KEY=$(echo "$MAXMIND_LICENSE_KEY" | tr -d '"' | tr -d "'" | tr -d '[:space:]')

  # Download the database using the sanitized key 
  curl -L -o /app/geoip_db/GeoLite2-Country.tar.gz \
    "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-Country&license_key=${MAXMIND_LICENSE_KEY}&suffix=tar.gz"

  # Extract the .mmdb file and remove archive
  tar -xzf /app/geoip_db/GeoLite2-Country.tar.gz -C /app/geoip_db --strip-components=1
  rm /app/geoip_db/GeoLite2-Country.tar.gz
  echo "Successfully downloaded GeoLite2-Country.mmdb."
fi

# Start the combined FastAPI and Gradio app on a uvicorn server (Hugging Face Spaces expects port 7860 even though default is 8000)
uvicorn backend.app:app --host 0.0.0.0 --port 7860
