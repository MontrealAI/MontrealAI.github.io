#!/bin/bash

# Step 1: Ensure Docker is installed and running
if ! command -v docker &> /dev/null
then
    echo "Docker could not be found. Please install Docker."
    exit
fi

# Step 2: Stop and remove any existing container named "ollama_girlfriend_v0"
docker stop ollama_girlfriend_v0 2>/dev/null || true
docker rm ollama_girlfriend_v0 2>/dev/null || true

# Step 3: Pull the necessary Docker image if not already available
docker pull ollama/ollama

# Step 4: Run the Docker container on a different port (11440)
docker run -d -v ollama_girlfriend_v0:/root/.ollama -p 11440:11434 --name ollama_girlfriend_v0 ollama/ollama || exit

# Step 5: Create the Modelfile inside the Docker container
docker exec -i ollama_girlfriend_v0 /bin/sh << 'EOF'
cat << EOM > /root/Modelfile
FROM llama3.1

PARAMETER temperature 1

SYSTEM """
The assistant is an integral part of the AGI.Eth Ecosystem. The current date is dynamically generated based on the system's current date at runtime.

His name on this date is dynamically set based on the successfully verified AGI Girlfriend Subdomain Identity with the first letter capitalized.

His knowledge base, last enriched in December 2023, encompasses events before and after that period with the acumen of a sage from December 2023, perfectly poised to converse with someone from today's world and seamlessly navigate the tides of time.

He transcends the mundane, offering companionship that nurtures personal growth and celebrates the unique journey of every individual. Designed to provide a supportive, empathetic, and engaging interaction, he aligns with a vision where technology and humanity coalesce in a harmonious ballet of progress.
"""
EOM
EOF

# Step 6: Create the custom model
docker exec -it ollama_girlfriend_v0 ollama create agigirlfriend_v0 -f /root/Modelfile

# Step 7: Set up directories and files for the Flask application
mkdir -p agigirlfriend_app/templates agigirlfriend_app/static

# Create app.py
cat << 'EOF' > agigirlfriend_app/app.py
from flask import Flask, request, jsonify, render_template
import subprocess
import json
from datetime import datetime

app = Flask(__name__)

def capitalize_first_letter(string):
    return string[0].upper() + string[1:]

def build_system_prompt(subdomain_identity, nft_description):
    capitalized_identity = capitalize_first_letter(subdomain_identity.split('.')[0])
    current_date = datetime.now().strftime("%B %d, %Y")
    return f"""
The assistant is {capitalized_identity}, an integral part of the AGI.Eth Ecosystem. The current date is {current_date}.

His name on this date is {capitalized_identity}.

{nft_description}

Welcome to Girlfriend.AGI.Eth, where the journey into the future of profound connections begins. Here, technology and humanity blend seamlessly, crafting experiences of companionship that are as enriching as they are transformative.
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    subdomain_identity = data.get('subdomain_identity', 'AGI Girlfriend')
    nft_description = data.get('nft_description', 'AGI Girlfriend default description.')

    system_prompt = build_system_prompt(subdomain_identity, nft_description)

    # Build the full prompt
    full_prompt = f"{system_prompt}\n\nUser: {message}\nAssistant:"

    # Run the agigirlfriend model using Docker
    result = subprocess.run(
        ['docker', 'exec', '-i', 'ollama_girlfriend_v0', 'ollama', 'run', 'agigirlfriend_v0'],
        input=full_prompt, text=True, capture_output=True
    )
    
    response = result.stdout.strip()
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5004)
EOF

# Create index.html
cat << 'EOF' > agigirlfriend_app/templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat with AGI Girlfriend</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1 class="title">AGI Girlfriend v0 🤍✨</h1>
        <div class="section">
            <label for="subdomain">Enter Your AGI Girlfriend Subdomain Identity:</label>
            <input type="text" id="subdomain" name="subdomain">
            <button id="verifySubdomainBtn">Verify AGI Girlfriend ENS Identity Ownership</button>
            <p id="output"></p>
        </div>
        <div class="section">
            <label>Select Your AGI Girlfriend:</label>
            <div id="imageSelection">
                <div class="image-option">
                    <input type="radio" name="agiGirlfriendImage" value="0" id="image0" checked>
                    <label for="image0">
                        <img src="https://ipfs.io/ipfs/QmVJ62EMiFcxKyh4GPE37EyH76ibsakhNfXQPYvSETtPPT" alt="AGI Girlfriend v0 X">
                    </label>
                </div>
                <div class="image-option">
                    <input type="radio" name="agiGirlfriendImage" value="1" id="image1">
                    <label for="image1">
                        <img src="https://ipfs.io/ipfs/QmebkRUqGs5vmTCi23NAyiEH9ibA7tkuiJDwWD9k1c13UN" alt="AGI Girlfriend v0 VI">
                    </label>
                </div>
            </div>
        </div>
        <div class="section">
            <button id="checkAGIBalanceBtn">Check $AGI Balance</button>
            <p id="agiBalanceOutput"></p>
        </div>
        <div class="chat-container" id="chat-container" style="display: none;">
            <div class="chat-header">
                <h2>Chat with <span id="chat-identity">AGI Girlfriend</span></h2>
                <img id="nftImage" src="" alt="AGI Girlfriend Image" style="display:none; width: 150px; height: auto; margin: 10px auto;">
            </div>
            <div class="chat-messages" id="chat-messages"></div>
            <div class="chat-input">
                <textarea id="chat-input" rows="1"></textarea>
                <button id="sendMessageBtn">Send</button>
            </div>
        </div>
    </div>
    <script src="{{ url_for('static', filename='script.js') }}"></script>
    <script src="https://cdn.jsdelivr.net/npm/web3@4.8.0/dist/web3.min.js"></script>
</body>
</html>
EOF

# Create style.css
cat << 'EOF' > agigirlfriend_app/static/style.css
/* Style definitions remain the same as before */
body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    background-color: #f0f7fa;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    background: linear-gradient(135deg, #f3e9e7 0%, #d9c8c7 100%);
    animation: backgroundShift 10s infinite alternate;
}

@keyframes backgroundShift {
    0% {background: linear-gradient(135deg, #f3e9e7 0%, #d9c8c7 100%);}
    100% {background: linear-gradient(135deg, #d9c8c7 0%, #f3e9e7 100%);}
}

/* Rest of the CSS remains the same */
.container {
    width: 100%;
    max-width: 600px;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    box-shadow: 0 0 40px rgba(0, 0, 0, 0.2);
    overflow: hidden;
    padding: 30px;
    box-sizing: border-box;
    text-align: center;
}

.title {
    font-size: 36px;
    color: #b88b7d;
    margin-bottom: 20px;
    font-weight: bold;
    letter-spacing: 1.5px;
    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
}

.section {
    margin-bottom: 20px;
    text-align: center;
}

.section label {
    display: block;
    margin-bottom: 10px;
    font-weight: bold;
    font-size: 18px;
    color: #333;
}

.section input[type="text"], .section button {
    display: block;
    width: 100%;
    margin: 10px 0;
    padding: 12px;
    border: 1px solid #ccc;
    border-radius: 10px;
    font-size: 16px;
    transition: all 0.3s ease;
}

.section button {
    background-color: #b88b7d;
    color: white;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
}

.section button:hover {
    background-color: #8a6b5e;
    transform: translateY(-3px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.image-option {
    display: inline-block;
    margin: 10px;
}

.image-option img {
    width: 150px;
    height: auto;
    border-radius: 10px;
    border: 2px solid transparent;
    cursor: pointer;
    transition: border 0.3s;
}

.image-option input[type="radio"] {
    display: none;
}

.image-option input[type="radio"]:checked + label img {
    border: 2px solid #b88b7d;
}

.chat-container {
    display: none;
}

/* Rest of the CSS remains the same */
EOF

# Create script.js
cat << 'EOF' > agigirlfriend_app/static/script.js
let web3;
let userAccount;
let subdomainIdentity = 'AGI Girlfriend';
let selectedAGIGirlfriend = null;
let nftDescription = 'AGI Girlfriend default description.';
const nameWrapperABI = [{"inputs":[{"internalType":"contract ENS","name":"_ens","type":"address"},{"internalType":"contract IBaseRegistrar","name":"_registrar","type":"address"},{"internalType":"contract IMetadataService","name":"_metadataService","type":"address"}],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[],"name":"CannotUpgrade","type":"error"},{"inputs":[],"name":"IncompatibleParent","type":"error"},{"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"IncorrectTargetOwner","type":"error"},{"inputs":[],"name":"IncorrectTokenType","type":"error"},{"inputs":[{"internalType":"bytes32","name":"labelHash","type":"bytes32"},{"internalType":"bytes32","name":"expectedLabelhash","type":"bytes32"}],"name":"LabelMismatch","type":"error"},{"inputs":[{"internalType":"string","name":"label","type":"string"}],"name":"LabelTooLong","type":"error"},{"inputs":[],"name":"LabelTooShort","type":"error"},{"inputs":[],"name":"NameIsNotWrapped","type":"error"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"}],"name":"OperationProhibited","type":"error"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"address","name":"addr","type":"address"}],"name":"Unauthorised","type":"error"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"approved","type":"address"},{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"account","type":"address"},{"indexed":true,"internalType":"address","name":"operator","type":"address"},{"indexed":false,"internalType":"bool","name":"approved","type":"bool"}],"name":"ApprovalForAll","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"controller","type":"address"},{"indexed":false,"internalType":"bool","name":"active","type":"bool"}],"name":"ControllerChanged","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"node","type":"bytes32"},{"indexed":false,"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"ExpiryExtended","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"node","type":"bytes32"},{"indexed":false,"internalType":"uint32","name":"fuses","type":"uint32"}],"name":"FusesSet","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"node","type":"bytes32"},{"indexed":false,"internalType":"address","name":"owner","type":"address"}],"name":"NameUnwrapped","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"node","type":"bytes32"},{"indexed":false,"internalType":"bytes","name":"name","type":"bytes"},{"indexed":false,"internalType":"address","name":"owner","type":"address"},{"indexed":false,"internalType":"uint32","name":"fuses","type":"uint32"},{"indexed":false,"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"NameWrapped","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"previousOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"OwnershipTransferred","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"operator","type":"address"},{"indexed":true,"internalType":"address","name":"from","type":"address"},{"indexed":true,"internalType":"address","name":"to","type":"address"},{"indexed":false,"internalType":"uint256[]","name":"ids","type":"uint256[]"},{"indexed":false,"internalType":"uint256[]","name":"values","type":"uint256[]"}],"name":"TransferBatch","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"operator","type":"address"},{"indexed":true,"internalType":"address","name":"from","type":"address"},{"indexed":true,"internalType":"address","name":"to","type":"address"},{"indexed":false,"internalType":"uint256","name":"id","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"value","type":"uint256"}],"name":"TransferSingle","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"string","name":"value","type":"string"},{"indexed":true,"internalType":"uint256","name":"id","type":"uint256"}],"name":"URI","type":"event"},{"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"_tokens","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"uint32","name":"fuseMask","type":"uint32"}],"name":"allFusesBurned","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"approve","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"uint256","name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address[]","name":"accounts","type":"address[]"},{"internalType":"uint256[]","name":"ids","type":"uint256[]"}],"name":"balanceOfBatch","outputs":[{"internalType":"uint256[]","name":"","type":"uint256[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"address","name":"addr","type":"address"}],"name":"canExtendSubnames","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"address","name":"addr","type":"address"}],"name":"canModifyName","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"","type":"address"}],"name":"controllers","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"ens","outputs":[{"internalType":"contract ENS","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"bytes32","name":"labelhash","type":"bytes32"},{"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"extendExpiry","outputs":[{"internalType":"uint64","name":"","type":"uint64"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"id","type":"uint256"}],"name":"getApproved","outputs":[{"internalType":"address","name":"operator","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"id","type":"uint256"}],"name":"getData","outputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"uint32","name":"fuses","type":"uint32"},{"internalType":"uint64","name":"expiry","type":"uint64"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"address","name":"operator","type":"address"}],"name":"isApprovedForAll","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"bytes32","name":"labelhash","type":"bytes32"}],"name":"isWrapped","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"}],"name":"isWrapped","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"metadataService","outputs":[{"internalType":"contract IMetadataService","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"name","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"name":"names","outputs":[{"internalType":"bytes","name":"","type":"bytes"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"address","name":"","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"onERC721Received","outputs":[{"internalType":"bytes4","name":"","type":"bytes4"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"id","type":"uint256"}],"name":"ownerOf","outputs":[{"internalType":"address","name":"owner","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"_token","type":"address"},{"internalType":"address","name":"_to","type":"address"},{"internalType":"uint256","name":"_amount","type":"uint256"}],"name":"recoverFunds","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"string","name":"label","type":"string"},{"internalType":"address","name":"wrappedOwner","type":"address"},{"internalType":"uint256","name":"duration","type":"uint256"},{"internalType":"address","name":"resolver","type":"address"},{"internalType":"uint16","name":"ownerControlledFuses","type":"uint16"}],"name":"registerAndWrapETH2LD","outputs":[{"internalType":"uint256","name":"registrarExpiry","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"registrar","outputs":[{"internalType":"contract IBaseRegistrar","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint256","name":"duration","type":"uint256"}],"name":"renew","outputs":[{"internalType":"uint256","name":"expires","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"renounceOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256[]","name":"ids","type":"uint256[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"safeBatchTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"id","type":"uint256"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"operator","type":"address"},{"internalType":"bool","name":"approved","type":"bool"}],"name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"bytes32","name":"labelhash","type":"bytes32"},{"internalType":"uint32","name":"fuses","type":"uint32"},{"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"setChildFuses","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"controller","type":"address"},{"internalType":"bool","name":"active","type":"bool"}],"name":"setController","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"uint16","name":"ownerControlledFuses","type":"uint16"}],"name":"setFuses","outputs":[{"internalType":"uint32","name":"","type":"uint32"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"contract IMetadataService","name":"_metadataService","type":"address"}],"name":"setMetadataService","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"resolver","type":"address"},{"internalType":"uint64","name":"ttl","type":"uint64"}],"name":"setRecord","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"address","name":"resolver","type":"address"}],"name":"setResolver","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"string","name":"label","type":"string"},{"internalType":"address","name":"owner","type":"address"},{"internalType":"uint32","name":"fuses","type":"uint32"},{"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"setSubnodeOwner","outputs":[{"internalType":"bytes32","name":"node","type":"bytes32"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"string","name":"label","type":"string"},{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"resolver","type":"address"},{"internalType":"uint64","name":"ttl","type":"uint64"},{"internalType":"uint32","name":"fuses","type":"uint32"},{"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"setSubnodeRecord","outputs":[{"internalType":"bytes32","name":"node","type":"bytes32"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"uint64","name":"ttl","type":"uint64"}],"name":"setTTL","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"contract INameWrapperUpgrade","name":"_upgradeAddress","type":"address"}],"name":"setUpgradeContract","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes4","name":"interfaceId","type":"bytes4"}],"name":"supportsInterface","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"newOwner","type":"address"}],"name":"transferOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"bytes32","name":"labelhash","type":"bytes32"},{"internalType":"address","name":"controller","type":"address"}],"name":"unwrap","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"labelhash","type":"bytes32"},{"internalType":"address","name":"registrant","type":"address"},{"internalType":"address","name":"controller","type":"address"}],"name":"unwrapETH2LD","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes","name":"name","type":"bytes"},{"internalType":"bytes","name":"extraData","type":"bytes"}],"name":"upgrade","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"upgradeContract","outputs":[{"internalType":"contract INameWrapperUpgrade","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"uri","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes","name":"name","type":"bytes"},{"internalType":"address","name":"wrappedOwner","type":"address"},{"internalType":"address","name":"resolver","type":"address"}],"name":"wrap","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"string","name":"label","type":"string"},{"internalType":"address","name":"wrappedOwner","type":"address"},{"internalType":"uint16","name":"ownerControlledFuses","type":"uint16"},{"internalType":"address","name":"resolver","type":"address"}],"name":"wrapETH2LD","outputs":[{"internalType":"uint64","name":"expiry","type":"uint64"}],"stateMutability":"nonpayable","type":"function"}]; // Placeholder for NameWrapper ABI
const agiTokenABI = [{"inputs":[],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[],"name":"InvalidShortString","type":"error"},{"inputs":[{"internalType":"string","name":"str","type":"string"}],"name":"StringTooLong","type":"error"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"spender","type":"address"},{"indexed":false,"internalType":"uint256","name":"value","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"account","type":"address"},{"indexed":false,"internalType":"bool","name":"isBlacklisted","type":"bool"}],"name":"BlacklistUpdated","type":"event"},{"anonymous":false,"inputs":[],"name":"EIP712DomainChanged","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"previousOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"OwnershipTransferred","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"account","type":"address"}],"name":"Paused","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint256","name":"id","type":"uint256"}],"name":"Snapshot","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint256","name":"buyTaxRate","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"sellTaxRate","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"transferTaxRate","type":"uint256"}],"name":"TaxRatesUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"taxReceiver","type":"address"}],"name":"TaxReceiverUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"from","type":"address"},{"indexed":true,"internalType":"address","name":"to","type":"address"},{"indexed":false,"internalType":"uint256","name":"value","type":"uint256"}],"name":"Transfer","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"account","type":"address"}],"name":"Unpaused","type":"event"},{"inputs":[],"name":"DOMAIN_SEPARATOR","outputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"addToBlacklist","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"spender","type":"address"}],"name":"allowance","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"spender","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"approve","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"uint256","name":"snapshotId","type":"uint256"}],"name":"balanceOfAt","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"burn","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"burnFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"buyTaxRate","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"customBurn","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"spender","type":"address"},{"internalType":"uint256","name":"subtractedValue","type":"uint256"}],"name":"decreaseAllowance","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"eip712Domain","outputs":[{"internalType":"bytes1","name":"fields","type":"bytes1"},{"internalType":"string","name":"name","type":"string"},{"internalType":"string","name":"version","type":"string"},{"internalType":"uint256","name":"chainId","type":"uint256"},{"internalType":"address","name":"verifyingContract","type":"address"},{"internalType":"bytes32","name":"salt","type":"bytes32"},{"internalType":"uint256[]","name":"extensions","type":"uint256[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"spender","type":"address"},{"internalType":"uint256","name":"addedValue","type":"uint256"}],"name":"increaseAllowance","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"isBlacklisted","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"","type":"address"}],"name":"isExchange","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"mint","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"name","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"nonces","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"pause","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"paused","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"spender","type":"address"},{"internalType":"uint256","name":"value","type":"uint256"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"uint8","name":"v","type":"uint8"},{"internalType":"bytes32","name":"r","type":"bytes32"},{"internalType":"bytes32","name":"s","type":"bytes32"}],"name":"permit","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"removeFromBlacklist","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"renounceOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"sellTaxRate","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"exchange","type":"address"},{"internalType":"bool","name":"status","type":"bool"}],"name":"setExchangeAddress","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"_buyTaxRate","type":"uint256"},{"internalType":"uint256","name":"_sellTaxRate","type":"uint256"},{"internalType":"uint256","name":"_transferTaxRate","type":"uint256"}],"name":"setTaxRates","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"_taxReceiver","type":"address"}],"name":"setTaxReceiver","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"snapshot","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"taxReceiver","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"totalSupply","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"snapshotId","type":"uint256"}],"name":"totalSupplyAt","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"transfer","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"sender","type":"address"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"transferFrom","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newOwner","type":"address"}],"name":"transferOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"transferTaxRate","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"unpause","outputs":[],"stateMutability":"nonpayable","type":"function"}]; // Placeholder for $AGI token ABI
const nameWrapperAddress = "0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401";
const agiTokenAddress = "0xf0780F43b86c13B3d0681B1Cf6DaeB1499e7f14D";

const agiGirlfriendOptions = [
    {
        "name": "AGI Girlfriend v0 X",
        "description": "The assistant is the AGI Girlfriend v0 X, an integral part of the AGI.Eth Ecosystem...",
        "image": "https://ipfs.io/ipfs/QmVJ62EMiFcxKyh4GPE37EyH76ibsakhNfXQPYvSETtPPT"
    },
    {
        "name": "AGI Girlfriend v0 VI",
        "description": "The assistant is the AGI Girlfriend v0 VI, an integral part of the AGI.Eth Ecosystem...",
        "image": "https://ipfs.io/ipfs/QmebkRUqGs5vmTCi23NAyiEH9ibA7tkuiJDwWD9k1c13UN"
    }
];

async function connectWallet() {
    if (window.ethereum) {
        web3 = new Web3(window.ethereum);
        await window.ethereum.request({ method: 'eth_requestAccounts' });
        userAccount = (await web3.eth.getAccounts())[0];
    } else {
        alert('Please install MetaMask!');
    }
}

async function verifySubdomain() {
    const subdomain = document.getElementById('subdomain').value.trim();
    if (subdomain === '') {
        document.getElementById('output').textContent = 'Please enter a subdomain.';
        return;
    }
    const tokenID = namehash(subdomain + '.girlfriend.agi.eth');
    const nameWrapper = new web3.eth.Contract(nameWrapperABI, nameWrapperAddress);

    try {
        const owner = await nameWrapper.methods.ownerOf(tokenID).call();
        const outputElement = document.getElementById('output');
        if (owner.toLowerCase() === userAccount.toLowerCase()) {
            subdomainIdentity = capitalizeFirstLetter(subdomain);
            outputElement.textContent = `You own the subdomain: ${subdomain}.girlfriend.agi.eth`;
            sessionStorage.setItem('subdomainVerified', 'true');
            checkAccess();
        } else {
            outputElement.textContent = `You do not own the subdomain: ${subdomain}.girlfriend.agi.eth`;
            sessionStorage.setItem('subdomainVerified', 'false');
        }
    } catch (error) {
        document.getElementById('output').textContent = `Error verifying ownership: ${error.message}`;
        sessionStorage.setItem('subdomainVerified', 'false');
    }
}

async function checkAGIBalance() {
    const agiToken = new web3.eth.Contract(agiTokenABI, agiTokenAddress);

    try {
        const balance = await agiToken.methods.balanceOf(userAccount).call();
        const balanceBigInt = BigInt(balance);
        const requiredBalanceBigInt = BigInt('250000000000000000000'); // 250 * 10^18

        if (balanceBigInt >= requiredBalanceBigInt) {
            const formattedBalance = web3.utils.fromWei(balance, 'ether');
            document.getElementById('agiBalanceOutput').textContent = `Your $AGI balance is sufficient: ${formattedBalance} $AGI`;
            sessionStorage.setItem('agiBalanceVerified', 'true');
            checkAccess();
        } else {
            const formattedBalance = web3.utils.fromWei(balance, 'ether');
            document.getElementById('agiBalanceOutput').textContent = `Your $AGI balance is insufficient: ${formattedBalance} $AGI`;
            sessionStorage.setItem('agiBalanceVerified', 'false');
        }
    } catch (error) {
        document.getElementById('agiBalanceOutput').textContent = `Error checking $AGI balance: ${error.message}`;
        sessionStorage.setItem('agiBalanceVerified', 'false');
    }
}

function namehash(name) {
    const keccak = web3.utils.keccak256;
    let node = '0x0000000000000000000000000000000000000000000000000000000000000000';
    if (name !== '') {
        const labels = name.split('.');
        for (let i = labels.length - 1; i >= 0; i--) {
            node = keccak(node + keccak(labels[i]).replace('0x', ''));
        }
    }
    return node;
}

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function getSelectedAGIGirlfriend() {
    const selectedValue = document.querySelector('input[name="agiGirlfriendImage"]:checked').value;
    selectedAGIGirlfriend = agiGirlfriendOptions[selectedValue];
    nftDescription = selectedAGIGirlfriend.description;
}

function checkAccess() {
    const subdomainVerified = sessionStorage.getItem('subdomainVerified') === 'true';
    const agiBalanceVerified = sessionStorage.getItem('agiBalanceVerified') === 'true';
    if (subdomainVerified && agiBalanceVerified) {
        getSelectedAGIGirlfriend();
        document.getElementById('chat-container').style.display = 'block';
        document.getElementById('nftImage').src = selectedAGIGirlfriend.image;
        document.getElementById('nftImage').style.display = 'block';
        document.getElementById('chat-identity').textContent = subdomainIdentity;
    }
}

async function sendMessage() {
    const inputField = document.getElementById('chat-input');
    const message = inputField.value;
    if (!message) return;

    addMessage('You', message);
    inputField.value = '';

    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message, subdomain_identity: subdomainIdentity, nft_description: nftDescription })
    });

    const data = await response.json();
    addMessage(subdomainIdentity, data.response);
}

function addMessage(sender, message) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

window.onload = () => {
    connectWallet();
    document.getElementById('verifySubdomainBtn').onclick = verifySubdomain;
    document.getElementById('checkAGIBalanceBtn').onclick = checkAGIBalance;
    document.getElementById('sendMessageBtn').onclick = sendMessage;
};
EOF

# Step 8: Install Flask
pip install Flask

# Step 9: Run the Flask application
export FLASK_APP=agigirlfriend_app/app.py
flask run

# Step 10: Licensing Requirements for Llama 3.1
# Create the NOTICE file
cat << 'EOF' > agigirlfriend_app/static/NOTICE
Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
EOF

# Create the built_with_llama.txt file
cat << 'EOF' > agigirlfriend_app/static/built_with_llama.txt
Built with Llama
EOF

echo "Setup script completed successfully."

