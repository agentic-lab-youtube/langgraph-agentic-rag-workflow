import os
from dotenv import load_dotenv

load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
EXCEL_PATH = os.path.join(PROJECT_ROOT, "data", "Incident_tickets_sample.xlsx")
