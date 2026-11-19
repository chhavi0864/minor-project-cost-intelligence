import os
from pathlib import Path
from sqlalchemy import (
    MetaData, Table, Column, String, Float, Text, Integer, create_engine
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = (BASE_DIR / "data" / "billing.db").as_posix()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)
metadata = MetaData()

billing = Table(
    "billing",
    metadata,
    Column("invoice_month", String, nullable=False),  # YYYY-MM
    Column("account_id", String),
    Column("subscription", String),
    Column("service", String),
    Column("resource_group", String),
    Column("resource_id", String),
    Column("region", String),
    Column("usage_qty", Float),
    Column("unit_cost", Float),
    Column("cost", Float),
)

resources = Table(
    "resources",
    metadata,
    Column("resource_id", String, primary_key=True),
    Column("owner", String),
    Column("env", String),
    Column("tags_json", Text),
)

# Standardized Universal FinOps Cost Records Table
cost_records = Table(
    "cost_records",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("import_batch_id", String, index=True),
    Column("billing_period", String, nullable=False, index=True),  # YYYY-MM
    Column("charge_period_start", String, index=True),             # ISO string or YYYY-MM-DD
    Column("charge_period_end", String),
    Column("cloud_provider", String, default="Unknown", index=True),
    Column("service_name", String, nullable=False, index=True),
    Column("service_category", String, nullable=False, index=True), # Compute, Database, Storage, Networking, Other
    Column("resource_id", String, index=True),
    Column("resource_name", String),
    Column("region", String),
    Column("usage_quantity", Float, default=0.0),
    Column("usage_unit", String),
    Column("billed_cost", Float, nullable=False),
    Column("currency", String, default="USD"),
    Column("tags_json", Text),
    Column("raw_metadata_json", Text),
)

# Ingestion Batches Table for Tracking and Duplicate Prevention
ingestion_batches = Table(
    "ingestion_batches",
    metadata,
    Column("batch_id", String, primary_key=True),
    Column("filename", String, nullable=False),
    Column("file_hash", String, nullable=False, index=True),  # SHA-256 hash
    Column("source_format", String, nullable=False),          # e.g. "FOCUS 1.0", "Legacy FinOps", "Custom"
    Column("total_records", Integer),
    Column("total_cost", Float),
    Column("currency", String),
    Column("date_range_start", String),
    Column("date_range_end", String),
    Column("status", String),
    Column("created_at", String),
)

def create_tables():
    metadata.create_all(engine)
    print("Tables created (if not existing). DB:", DATABASE_URL)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    create_tables()
