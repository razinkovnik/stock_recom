import os
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), 'settings.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)