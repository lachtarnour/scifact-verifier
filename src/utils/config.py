import os
from pathlib import Path
from dataclasses import dataclass, field 

BASE_DIR = Path(__file__).resolve().parent.parent.parent

@dataclass
class config:
    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    INDEX_DIR: Path = BASE_DIR / "data" / "index"   
    REPORTS_DIR: Path = BASE_DIR / "reports"    
