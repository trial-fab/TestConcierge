# Setup Guide

## Prerequisites
- Python 3.10+
- Pip package manager
- Active Azure OpenAI account
- Supabase account (for vector database)
- Hugging Face API token
- Google Workspace credentials (optional, for email/calendar features)

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd TA-test
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys and configuration values:
- **Azure OpenAI**: Add your endpoint, API key, and deployment names
- **Hugging Face**: Add your API token for embeddings
- **Supabase**: Add your URL and service role key
- **Google Workspace**: Add OAuth credentials (if using email/meeting assistants)

### 4. Initialize Vector Database
```bash
python setup_db.py
```

This will:
- Connect to your Supabase instance
- Create necessary tables (documents, chunks, sessions, messages, users, audit_logs)
- Ingest documents from `data/raw/` into the vector database

### 5. Run the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Troubleshooting

### Common Issues

**Import Errors**: Make sure you've installed all dependencies with `pip install -r requirements.txt`

**Database Connection Errors**: Verify your `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in `.env`

**Azure OpenAI Errors**: Ensure your deployment names match those in your Azure OpenAI resource

**Google Workspace Errors**: Check that your OAuth credentials are valid and have the necessary scopes

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Supabase Documentation](https://supabase.com/docs)
- [Hugging Face Documentation](https://huggingface.co/docs)
