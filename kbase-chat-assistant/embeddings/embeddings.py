from pathlib import Path

DEFAULT_CATALOG_DB_DIR: Path = Path(__file__).parent / "vector_db_app_catalog"
DEFAULT_DOCS_DB_DIR: Path = Path(__file__).parent / "vector_db_kbase_docs"

HF_CATALOG_DB_DIR: Path = Path(__file__).parent / "HFvector_db_app_catalog"
HF_DOCS_DB_DIR: Path = Path(__file__).parent / "HFvector_db_kbase_docs"
