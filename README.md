# Financial Health Checker Agent

Playing with observability platforms

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages listed in `requirements.txt`

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   ```
5. Run the application or open notebooks in Jupyter:
   ```bash
   streamlit run app.py
   ```