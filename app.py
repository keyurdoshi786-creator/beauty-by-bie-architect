"""
AURA: The BiE Brand Architect


# Create and activate a virtualenv
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key (either method):
#   Option A: .env file
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
#   Option B: Streamlit secrets
mkdir -p .streamlit && echo 'ANTHROPIC_API_KEY = "sk-ant-..."' > .streamlit/secrets.toml

# Launch
streamlit run app.py
