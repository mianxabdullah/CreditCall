from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL) # Create a database engine that means we can connect to the database and execute SQL queries. The engine is created using the connection string stored in the DATABASE_URL environment variable.