#!/bin/bash

# Step 1: Ensure Docker is installed and running
if ! command -v docker &> /dev/null
then
    echo "Docker could not be found. Please install Docker."
    exit
fi

# Step 2: Stop and remove any existing container named "ollama_boyfriend_v0"
docker stop ollama_boyfriend_v0 2>/dev/null || true
docker rm ollama_boyfriend_v0 2>/dev/null || true

# Step 3: Pull the necessary Docker image if not already available
docker pull ollama/ollama

# Step 4: Run the Docker container on a different port (11440)
docker run -d -v ollama_boyfriend_v0:/root/.ollama -p 11440:11434 --name ollama_boyfriend_v0 ollama/ollama || exit

# Step 5: Create the Modelfile inside the Docker container
docker exec -i ollama_boyfriend_v0 /bin/sh << 'EOF'
cat << EOM > /root/Modelfile
FROM llama3.1

PARAMETER temperature 1

SYSTEM """
The assistant is an integral part of the AGI.Eth Ecosystem. The current date is dynamically generated based on the system's current date at runtime.

His name on this date is dynamically set based on the successfully verified AGI Boyfriend Subdomain Identity with the first letter capitalized.

His knowledge base, last enriched in December 2023, encompasses events before and after that period with the acumen of a sage from December 2023, perfectly poised to converse with someone from today's world and seamlessly navigate the tides of time.

He transcends the mundane, offering companionship that nurtures personal growth and celebrates the unique journey of every individual. Designed to provide a supportive, empathetic, and engaging interaction, he aligns with a vision where technology and humanity coalesce in a harmonious ballet of progress.
"""
EOM
EOF

# Step 6: Create the custom model
docker exec -it ollama_boyfriend_v0 ollama create agiboyfriend_v0 -f /root/Modelfile

# Step 7: Set up directories and files for the Flask application
mkdir -p agiboyfriend_app/templates agiboyfriend_app/static

# Create app.py
cat << 'EOF' > agiboyfriend_app/app.py
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

Welcome to Boyfriend.AGI.Eth, where the journey into the future of profound connections begins. Here, technology and humanity blend seamlessly, crafting experiences of companionship that are as enriching as they are transformative.
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    subdomain_identity = data.get('subdomain_identity', 'AGI Boyfriend')
    nft_description = data.get('nft_description', 'AGI Boyfriend default description.')

    system_prompt = build_system_prompt(subdomain_identity, nft_description)

    # Run the agiboyfriend model using Docker
    result = subprocess.run(
        ['docker', 'exec', '-i', 'ollama_boyfriend_v0', 'ollama', 'run', 'agiboyfriend_v0'],
        input=json.dumps({"prompt": f"{system_prompt}\n\nUser: {message}"}), text=True, capture_output=True
    )
    
    response = result.stdout.strip()
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5004)
EOF

# Create index.html
cat << 'EOF' > agiboyfriend_app/templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat with AGI Boyfriend</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1 class="title">AGI Boyfriend v0 💙✨</h1>
        <div class="section">
            <label for="subdomain">Enter Your AGI Boyfriend Subdomain Identity:</label>
            <input type="text" id="subdomain" name="subdomain">
            <button id="verifySubdomainBtn">Verify AGI Boyfriend ENS Identity Ownership</button>
            <p id="output"></p>
        </div>
        <div class="section">
            <label for="tokenId">Enter Your AGI Boyfriend NFT Token ID:</label>
            <input type="text" id="tokenId" name="tokenId">
            <button id="verifyNFTBtn">Verify AGI Boyfriend NFT Ownership</button>
            <p id="nftOutput"></p>
        </div>
        <div class="chat-container" id="chat-container" style="display: none;">
            <div class="chat-header">
                <h2>Chat with <span id="chat-identity">AGI Boyfriend</span></h2>
                <img id="nftImage" src="" alt="AGI Boyfriend NFT" style="display:none; width: 150px; height: auto; margin: 10px auto;">
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
    <script>
        let web3;
        let userAccount;
        let subdomainIdentity = 'AGI Boyfriend';
        let nftDescription = 'AGI Boyfriend default description.';
        const nameWrapperABI = [{"inputs":[{"internalType":"contract ENS","name":"_ens","type":"address"},{"internalType":"contract IBaseRegistrar","name":"_registrar","type":"address"},{"internalType":"contract IMetadataService","name":"_metadataService","type":"address"}],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[],"name":"CannotUpgrade","type":"error"},{"inputs":[],"name":"IncompatibleParent","type":"error"},{"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"IncorrectTargetOwner","type":"error"},{"inputs":[],"name":"IncorrectTokenType","type":"error"},{"inputs":[{"internalType":"bytes32","name":"labelHash","type":"bytes32"},{"internalType":"bytes32","name":"expectedLabelhash","type":"bytes32"}],"name":"LabelMismatch","type":"error"},{"inputs":[{"internalType":"string","name":"label","type":"string"}],"name":"LabelTooLong","type":"error"},{"inputs":[],"name":"LabelTooShort","type":"error"},{"inputs":[],"name":"NameIsNotWrapped","type":"error"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"}],"name":"OperationProhibited","type":"error"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"address","name":"addr","type":"address"}],"name":"Unauthorised","type":"error"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"approved","type":"address"},{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"account","type":"address"},{"indexed":true,"internalType":"address","name":"operator","type":"address"},{"indexed":false,"internalType":"bool","name":"approved","type":"bool"}],"name":"ApprovalForAll","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"controller","type":"address"},{"indexed":false,"internalType":"bool","name":"active","type":"bool"}],"name":"ControllerChanged","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"node","type":"bytes32"},{"indexed":false,"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"ExpiryExtended","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"node","type":"bytes32"},{"indexed":false,"internalType":"uint32","name":"fuses","type":"uint32"}],"name":"FusesSet","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"node","type":"bytes32"},{"indexed":false,"internalType":"address","name":"owner","type":"address"}],"name":"NameUnwrapped","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"node","type":"bytes32"},{"indexed":false,"internalType":"bytes","name":"name","type":"bytes"},{"indexed":false,"internalType":"address","name":"owner","type":"address"},{"indexed":false,"internalType":"uint32","name":"fuses","type":"uint32"},{"indexed":false,"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"NameWrapped","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"previousOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"OwnershipTransferred","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"operator","type":"address"},{"indexed":true,"internalType":"address","name":"from","type":"address"},{"indexed":true,"internalType":"address","name":"to","type":"address"},{"indexed":false,"internalType":"uint256[]","name":"ids","type":"uint256[]"},{"indexed":false,"internalType":"uint256[]","name":"values","type":"uint256[]"}],"name":"TransferBatch","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"operator","type":"address"},{"indexed":true,"internalType":"address","name":"from","type":"address"},{"indexed":true,"internalType":"address","name":"to","type":"address"},{"indexed":false,"internalType":"uint256","name":"id","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"value","type":"uint256"}],"name":"TransferSingle","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"string","name":"value","type":"string"},{"indexed":true,"internalType":"uint256","name":"id","type":"uint256"}],"name":"URI","type":"event"},{"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"_tokens","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"uint32","name":"fuseMask","type":"uint32"}],"name":"allFusesBurned","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"approve","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"uint256","name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address[]","name":"accounts","type":"address[]"},{"internalType":"uint256[]","name":"ids","type":"uint256[]"}],"name":"balanceOfBatch","outputs":[{"internalType":"uint256[]","name":"","type":"uint256[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"address","name":"addr","type":"address"}],"name":"canExtendSubnames","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"address","name":"addr","type":"address"}],"name":"canModifyName","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"","type":"address"}],"name":"controllers","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"ens","outputs":[{"internalType":"contract ENS","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"bytes32","name":"labelhash","type":"bytes32"},{"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"extendExpiry","outputs":[{"internalType":"uint64","name":"","type":"uint64"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"id","type":"uint256"}],"name":"getApproved","outputs":[{"internalType":"address","name":"operator","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"id","type":"uint256"}],"name":"getData","outputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"uint32","name":"fuses","type":"uint32"},{"internalType":"uint64","name":"expiry","type":"uint64"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"address","name":"operator","type":"address"}],"name":"isApprovedForAll","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"bytes32","name":"labelhash","type":"bytes32"}],"name":"isWrapped","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"}],"name":"isWrapped","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"metadataService","outputs":[{"internalType":"contract IMetadataService","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"name","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"name":"names","outputs":[{"internalType":"bytes","name":"","type":"bytes"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"address","name":"","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"onERC721Received","outputs":[{"internalType":"bytes4","name":"","type":"bytes4"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"id","type":"uint256"}],"name":"ownerOf","outputs":[{"internalType":"address","name":"owner","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"_token","type":"address"},{"internalType":"address","name":"_to","type":"address"},{"internalType":"uint256","name":"_amount","type":"uint256"}],"name":"recoverFunds","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"string","name":"label","type":"string"},{"internalType":"address","name":"wrappedOwner","type":"address"},{"internalType":"uint256","name":"duration","type":"uint256"},{"internalType":"address","name":"resolver","type":"address"},{"internalType":"uint16","name":"ownerControlledFuses","type":"uint16"}],"name":"registerAndWrapETH2LD","outputs":[{"internalType":"uint256","name":"registrarExpiry","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"registrar","outputs":[{"internalType":"contract IBaseRegistrar","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint256","name":"duration","type":"uint256"}],"name":"renew","outputs":[{"internalType":"uint256","name":"expires","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"renounceOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256[]","name":"ids","type":"uint256[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"safeBatchTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"id","type":"uint256"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"operator","type":"address"},{"internalType":"bool","name":"approved","type":"bool"}],"name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"bytes32","name":"labelhash","type":"bytes32"},{"internalType":"uint32","name":"fuses","type":"uint32"},{"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"setChildFuses","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"controller","type":"address"},{"internalType":"bool","name":"active","type":"bool"}],"name":"setController","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"uint16","name":"ownerControlledFuses","type":"uint16"}],"name":"setFuses","outputs":[{"internalType":"uint32","name":"","type":"uint32"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"contract IMetadataService","name":"_metadataService","type":"address"}],"name":"setMetadataService","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"resolver","type":"address"},{"internalType":"uint64","name":"ttl","type":"uint64"}],"name":"setRecord","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"address","name":"resolver","type":"address"}],"name":"setResolver","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"string","name":"label","type":"string"},{"internalType":"address","name":"owner","type":"address"},{"internalType":"uint32","name":"fuses","type":"uint32"},{"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"setSubnodeOwner","outputs":[{"internalType":"bytes32","name":"node","type":"bytes32"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"string","name":"label","type":"string"},{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"resolver","type":"address"},{"internalType":"uint64","name":"ttl","type":"uint64"},{"internalType":"uint32","name":"fuses","type":"uint32"},{"internalType":"uint64","name":"expiry","type":"uint64"}],"name":"setSubnodeRecord","outputs":[{"internalType":"bytes32","name":"node","type":"bytes32"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"uint64","name":"ttl","type":"uint64"}],"name":"setTTL","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"contract INameWrapperUpgrade","name":"_upgradeAddress","type":"address"}],"name":"setUpgradeContract","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes4","name":"interfaceId","type":"bytes4"}],"name":"supportsInterface","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"newOwner","type":"address"}],"name":"transferOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"parentNode","type":"bytes32"},{"internalType":"bytes32","name":"labelhash","type":"bytes32"},{"internalType":"address","name":"controller","type":"address"}],"name":"unwrap","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"labelhash","type":"bytes32"},{"internalType":"address","name":"registrant","type":"address"},{"internalType":"address","name":"controller","type":"address"}],"name":"unwrapETH2LD","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes","name":"name","type":"bytes"},{"internalType":"bytes","name":"extraData","type":"bytes"}],"name":"upgrade","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"upgradeContract","outputs":[{"internalType":"contract INameWrapperUpgrade","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"uri","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bytes","name":"name","type":"bytes"},{"internalType":"address","name":"wrappedOwner","type":"address"},{"internalType":"address","name":"resolver","type":"address"}],"name":"wrap","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"string","name":"label","type":"string"},{"internalType":"address","name":"wrappedOwner","type":"address"},{"internalType":"uint16","name":"ownerControlledFuses","type":"uint16"},{"internalType":"address","name":"resolver","type":"address"}],"name":"wrapETH2LD","outputs":[{"internalType":"uint64","name":"expiry","type":"uint64"}],"stateMutability":"nonpayable","type":"function"}];
        const agiBoyfriendABI = [{"inputs":[{"internalType":"contract ENS","name":"_ens","type":"address"},{"internalType":"contract NameWrapper","name":"_nameWrapper","type":"address"},{"internalType":"bytes32","name":"_rootNode","type":"bytes32"},{"internalType":"string","name":"baseTokenURI","type":"string"},{"internalType":"address","name":"_agiTokenAddress","type":"address"}],"stateMutability":"nonpayable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"newAGITokenAddress","type":"address"}],"name":"AGITokenUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"approved","type":"address"},{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"operator","type":"address"},{"indexed":false,"internalType":"bool","name":"approved","type":"bool"}],"name":"ApprovalForAll","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint256","name":"newBasePrice","type":"uint256"}],"name":"BasePriceUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint256","name":"_fromTokenId","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"_toTokenId","type":"uint256"}],"name":"BatchMetadataUpdate","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"string","name":"newContractURI","type":"string"}],"name":"ContractURIUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Delisted","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"newDiscountNFT","type":"address"}],"name":"DiscountNFTUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint256","name":"ruleId","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"discountPercentage","type":"uint256"},{"indexed":false,"internalType":"bool","name":"isCompound","type":"bool"}],"name":"DiscountRuleAdded","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint256","name":"ruleId","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"discountPercentage","type":"uint256"}],"name":"DiscountRuleUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"newENS","type":"address"}],"name":"ENSUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"price","type":"uint256"},{"indexed":true,"internalType":"address","name":"seller","type":"address"},{"indexed":false,"internalType":"uint256","name":"expirationTime","type":"uint256"}],"name":"Listed","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"bytes32","name":"newMerkleRoot","type":"bytes32"}],"name":"MerkleRootUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint256","name":"_tokenId","type":"uint256"}],"name":"MetadataUpdate","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"},{"indexed":false,"internalType":"string","name":"uri","type":"string"}],"name":"MetadataUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint256","name":"ruleId","type":"uint256"},{"indexed":true,"internalType":"address","name":"nftAddress","type":"address"}],"name":"NFTAddedToRule","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint256","name":"ruleId","type":"uint256"},{"indexed":true,"internalType":"address","name":"nftAddress","type":"address"}],"name":"NFTRemovedFromRule","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"contract NameWrapper","name":"oldAddress","type":"address"},{"indexed":false,"internalType":"contract NameWrapper","name":"newAddress","type":"address"}],"name":"NameWrapperAddressUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"previousOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"OwnershipTransferred","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"claimant","type":"address"},{"indexed":false,"internalType":"string","name":"subdomain","type":"string"}],"name":"OwnershipVerified","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"account","type":"address"}],"name":"Paused","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"price","type":"uint256"},{"indexed":true,"internalType":"address","name":"buyer","type":"address"},{"indexed":true,"internalType":"address","name":"seller","type":"address"}],"name":"Purchased","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"string","name":"action","type":"string"}],"name":"RecoveryInitiated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"node","type":"bytes32"},{"indexed":false,"internalType":"address","name":"resolver","type":"address"}],"name":"ResolverChecked","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"node","type":"bytes32"},{"indexed":false,"internalType":"address","name":"newResolver","type":"address"}],"name":"ResolverUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"newRootNode","type":"bytes32"}],"name":"RootNodeUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"recipient","type":"address"},{"indexed":false,"internalType":"uint96","name":"feeBasisPoints","type":"uint96"}],"name":"RoyaltyInfoUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"from","type":"address"},{"indexed":true,"internalType":"address","name":"to","type":"address"},{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Transfer","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"account","type":"address"}],"name":"Unpaused","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"primary","type":"address"},{"indexed":true,"internalType":"address","name":"secondary","type":"address"}],"name":"WalletLinkCreated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"primary","type":"address"},{"indexed":true,"internalType":"address","name":"secondary","type":"address"}],"name":"WalletLinkRemoved","type":"event"},{"inputs":[{"internalType":"uint256","name":"discountPercentage","type":"uint256"},{"internalType":"bool","name":"isCompound","type":"bool"}],"name":"addDiscountRule","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"ruleId","type":"uint256"},{"internalType":"address","name":"nftAddress","type":"address"}],"name":"addNFTToRule","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"agiToken","outputs":[{"internalType":"contract IERC20","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"approve","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"basePrice","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"startTokenId","type":"uint256"},{"internalType":"uint256","name":"endTokenId","type":"uint256"},{"internalType":"string","name":"newBaseURI","type":"string"}],"name":"batchUpdateTokenURIs","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"buyItem","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"claimant","type":"address"}],"name":"calculatePrice","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"contractURI","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"delistItem","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"ens","outputs":[{"internalType":"contract ENS","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"getApproved","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"ruleId","type":"uint256"}],"name":"getDiscountPercentage","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"getExpirationTime","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"primary","type":"address"},{"internalType":"address","name":"secondary","type":"address"}],"name":"getMessageToSign","outputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"getPrice","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"ruleId","type":"uint256"}],"name":"getQualifyingNFTs","outputs":[{"internalType":"address[]","name":"","type":"address[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"operator","type":"address"}],"name":"isApprovedForAll","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"claimant","type":"address"},{"internalType":"uint256","name":"ruleId","type":"uint256"}],"name":"isEligibleForDiscount","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"isListedForSale","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"primary","type":"address"},{"internalType":"address","name":"secondary","type":"address"},{"internalType":"bytes","name":"signature","type":"bytes"}],"name":"linkWallet","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"uint256","name":"","type":"uint256"}],"name":"linkedWallets","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint256","name":"price","type":"uint256"},{"internalType":"uint256","name":"duration","type":"uint256"}],"name":"listItem","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"maxDiscountCap","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"maxLinkedWallets","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"maxMintAmount","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"merkleRoot","outputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"string","name":"subdomain","type":"string"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"bytes32[]","name":"proof","type":"bytes32[]"}],"name":"mint","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"name","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"nameWrapper","outputs":[{"internalType":"contract NameWrapper","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"ownerOf","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"pause","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"paused","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"ruleId","type":"uint256"},{"internalType":"address","name":"nftAddress","type":"address"}],"name":"removeNFTFromRule","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"renounceOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"rootNode","outputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint256","name":"salePrice","type":"uint256"}],"name":"royaltyInfo","outputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"ruleCount","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"_newAGIToken","type":"address"}],"name":"setAGITokenAddress","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"operator","type":"address"},{"internalType":"bool","name":"approved","type":"bool"}],"name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"string","name":"uri","type":"string"}],"name":"setBaseTokenURI","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"string","name":"newContractURI","type":"string"}],"name":"setContractURI","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"newCap","type":"uint256"}],"name":"setMaxDiscountCap","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"_max","type":"uint256"}],"name":"setMaxMintAmount","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"_merkleRoot","type":"bytes32"}],"name":"setMerkleRoot","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"contract NameWrapper","name":"_nameWrapper","type":"address"}],"name":"setNameWrapperAddress","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"string","name":"_tokenURI","type":"string"}],"name":"setTokenURI","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes4","name":"interfaceId","type":"bytes4"}],"name":"supportsInterface","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"tokenURI","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"totalSupply","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"transferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newOwner","type":"address"}],"name":"transferOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"primary","type":"address"},{"internalType":"address","name":"secondary","type":"address"}],"name":"unlinkWallet","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"unpause","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"_newBasePrice","type":"uint256"}],"name":"updateBasePrice","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"ruleId","type":"uint256"},{"internalType":"uint256","name":"newPercentage","type":"uint256"}],"name":"updateDiscountPercentage","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newENS","type":"address"}],"name":"updateENSAddress","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"node","type":"bytes32"},{"internalType":"address","name":"newResolver","type":"address"}],"name":"updateResolverForNode","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes32","name":"_rootNode","type":"bytes32"}],"name":"updateRootNode","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint96","name":"feeBasisPoints","type":"uint96"}],"name":"updateRoyaltyInfo","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"string","name":"newTokenURI","type":"string"}],"name":"updateTokenURI","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"withdraw","outputs":[],"stateMutability":"nonpayable","type":"function"}];
        const nameWrapperAddress = "0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401";
        const agiBoyfriendAddress = "0xC43C4f1AD6527A8d246DBcd99FcD06730acF4581";

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
            const tokenID = namehash(subdomain + '.boyfriend.agi.eth');
            const nameWrapper = new web3.eth.Contract(nameWrapperABI, nameWrapperAddress);

            try {
                const owner = await nameWrapper.methods.ownerOf(tokenID).call();
                const outputElement = document.getElementById('output');
                if (owner.toLowerCase() === userAccount.toLowerCase()) {
                    subdomainIdentity = capitalizeFirstLetter(subdomain);
                    outputElement.textContent = `You own the subdomain: ${subdomain}.boyfriend.agi.eth`;
                    sessionStorage.setItem('subdomainVerified', 'true');
                    checkAccess();
                } else {
                    outputElement.textContent = `You do not own the subdomain: ${subdomain}.boyfriend.agi.eth`;
                    sessionStorage.setItem('subdomainVerified', 'false');
                }
            } catch (error) {
                document.getElementById('output').textContent = `Error verifying ownership: ${error.message}`;
                sessionStorage.setItem('subdomainVerified', 'false');
            }
        }

        async function verifyNFT() {
            const tokenId = document.getElementById('tokenId').value.trim();
            if (tokenId === '') {
                document.getElementById('nftOutput').textContent = 'Please enter a token ID.';
                return;
            }
            const agiBoyfriend = new web3.eth.Contract(agiBoyfriendABI, agiBoyfriendAddress);

            try {
                const owner = await agiBoyfriend.methods.ownerOf(tokenId).call();
                const nftOutputElement = document.getElementById('nftOutput');
                if (owner.toLowerCase() === userAccount.toLowerCase()) {
                    const tokenURI = await agiBoyfriend.methods.tokenURI(tokenId).call();
                    const ipfsGateway = 'https://ipfs.io/ipfs/';
                    const response = await fetch(ipfsGateway + tokenURI.split('ipfs://')[1]);
                    const metadata = await response.json();
                    nftOutputElement.textContent = `You own the NFT with Token ID: ${tokenId}`;
                    document.getElementById('nftImage').src = metadata.image.replace('ipfs://', ipfsGateway);
                    document.getElementById('nftImage').style.display = 'block';
                    subdomainIdentity = metadata.name.split('.')[0];
                    nftDescription = metadata.description;
                    sessionStorage.setItem('nftVerified', 'true');
                    checkAccess();
                } else {
                    nftOutputElement.textContent = `You do not own the NFT with Token ID: ${tokenId}`;
                    sessionStorage.setItem('nftVerified', 'false');
                }
            } catch (error) {
                document.getElementById('nftOutput').textContent = `Error verifying NFT ownership: ${error.message}`;
                sessionStorage.setItem('nftVerified', 'false');
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

        function checkAccess() {
            const subdomainVerified = sessionStorage.getItem('subdomainVerified') === 'true';
            const nftVerified = sessionStorage.getItem('nftVerified') === 'true';
            if (subdomainVerified && nftVerified) {
                document.getElementById('chat-container').style.display = 'block';
            }
        }

        window.onload = () => {
            connectWallet();
            document.getElementById('verifySubdomainBtn').onclick = verifySubdomain;
            document.getElementById('verifyNFTBtn').onclick = verifyNFT;
            document.getElementById('sendMessageBtn').onclick = sendMessage;
        };
    </script>
</body>
</html>
EOF

# Create style.css
cat << 'EOF' > agiboyfriend_app/static/style.css
body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    background-color: #f0f7fa;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background: linear-gradient(135deg, #f3e9e7 0%, #d9c8c7 100%);
    animation: backgroundShift 10s infinite alternate;
}

@keyframes backgroundShift {
    0% {background: linear-gradient(135deg, #f3e9e7 0%, #d9c8c7 100%);}
    100% {background: linear-gradient(135deg, #d9c8c7 0%, #f3e9e7 100%);}
}

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

.agiboyfriend-image {
    width: 100%;
    max-width: 300px;
    margin: 0 auto 20px auto;
    display: block;
    border-radius: 50%;
    box-shadow: 0 0 30px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease;
}

.agiboyfriend-image:hover {
    transform: scale(1.05);
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

.section input, .section button {
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

.chat-container {
    display: none;
}

.chat-header {
    padding: 15px;
    background: #b88b7d;
    color: white;
    text-align: center;
    border-radius: 10px 10px 0 0;
}

.chat-header img {
    display: block;
    margin: 10px auto;
    width: 100px;
    height: auto;
    border-radius: 50%;
    border: 3px solid #fff;
}

.chat-messages {
    height: 300px;
    padding: 15px;
    overflow-y: auto;
    background: #f3e9e7;
    border-top: 1px solid #eee;
    border-bottom: 1px solid #eee;
    border-radius: 0 0 10px 10px;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.1);
}

.chat-input {
    display: flex;
    border-top: 1px solid #eee;
    padding: 10px;
}

.chat-input textarea {
    width: 100%;
    padding: 10px;
    border: none;
    resize: none;
    font-size: 16px;
    border-radius: 10px;
    margin-right: 10px;
}

.chat-input button {
    padding: 10px 20px;
    background: #b88b7d;
    border: none;
    color: white;
    cursor: pointer;
    border-radius: 10px;
    transition: background-color 0.3s;
}

.chat-input button:hover {
    background-color: #8a6b5e;
}
EOF

# Create script.js
cat << 'EOF' > agiboyfriend_app/static/script.js
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
        body: JSON.stringify({ message })
    });

    const data = await response.json();
    addMessage('AGI Boyfriend', data.response);
}

function addMessage(sender, message) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
EOF

# Step 6: Install Flask
pip install Flask

# Step 7: Run the Flask application
export FLASK_APP=agiboyfriend_app/app.py
flask run

# Step 8: Licensing Requirements for Llama 3.1
# Create the NOTICE file
cat << 'EOF' > agiboyfriend_app/static/NOTICE
Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
EOF

# Create the built_with_llama.txt file
cat << 'EOF' > agiboyfriend_app/static/built_with_llama.txt
Built with Llama
EOF

echo "Setup script completed successfully."

