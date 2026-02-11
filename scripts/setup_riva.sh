#!/bin/bash
set -e
source .env
export PATH=$PATH:~/.local/ngc-cli

echo "Setting up Riva Models..."

# 1. Install NGC CLI (if missing)
if ! command -v ngc &> /dev/null; then
    echo "Installing NGC CLI..."
    wget -qO ngccli_arm64.zip https://ngc.nvidia.com/downloads/ngccli_arm64.zip
    unzip -o ngccli_arm64.zip
    chmod +x ngc-cli/ngc
    rm -rf ~/.local/ngc-cli
    rm -f ~/.local/bin/ngc # Cleanup stale symlink
    mv ngc-cli ~/.local/ngc-cli
    export PATH=$PATH:~/.local/ngc-cli
    rm ngccli_arm64.zip
else
    echo "NGC CLI found."
fi

# 2. Configure NGC
echo "Checking NGC Configuration..."
# User must have run `ngc config set` manually.
# To set up: ~/.local/ngc-cli/ngc config set
# Select API Key, JSON format, and your Org.

# Verify
ngc diag all || { echo "NGC improperly configured. Please run '~/.local/ngc-cli/ngc config set'"; exit 1; }

# 3. Download Riva Quickstart (ARM64)
# 3. Download Riva Quickstart (ARM64)
echo "Downloading Riva Quickstart..."
# Note: Version must match or be compatible. 2.24.0 is for JetPack 7 (Thor).
ngc registry resource download-version "nvidia/riva/riva_quickstart_arm64:2.24.0" --dest . || true

cd riva_quickstart_arm64_v2.24.0

# 4. Modify config.sh for Parakeet
echo "Configuring Riva for Parakeet..."
sed -i 's/service_enabled_asr=true/service_enabled_asr=true/g' config.sh
sed -i 's/service_enabled_nlp=true/service_enabled_nlp=true/g' config.sh
sed -i 's/service_enabled_tts=true/service_enabled_tts=true/g' config.sh # We use CosyVoice
sed -i 's/riva_model_architecture="kaldlm"/riva_model_architecture="ar_punc"/g' config.sh
# Check if EULA needs manual accepting here too
sed -i 's/# RIVA_EULA=accept/RIVA_EULA=accept/g' config.sh

# Set Output Directory
OUTPUT_REPO="$(pwd)/../../riva_model_repo_2.24"
mkdir -p "$OUTPUT_REPO"
sed -i "s|riva_model_repo=\".*\"|riva_model_repo=\"$OUTPUT_REPO\"|g" config.sh

# Uncomment Parakeet (Check if syntax is same in 2.24 config)
sed -i 's/#.*RMIR_ASR_PARAKEET_TDT_1_1B.*/RMIR_ASR_PARAKEET_TDT_1_1B="nvidia\/riva\/rmir_asr_parakeet_tdt_1.1b_en_us:${RIVA_NGC_VERSION}"/g' config.sh

# 5. Run Init
echo "Initializing Riva (Downloading Models)..."
bash riva_init.sh

echo "Riva Setup Complete. Models are in $OUTPUT_REPO"
