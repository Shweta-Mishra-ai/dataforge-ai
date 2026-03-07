import io
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from core.data_profiler import DatasetProfile


@dataclass
class ReportConfig:
    title: str = "Data Analysis Report"
    client_name: str = "Client"
    prepared_by: str = "DataForge AI"
    confidential: bool = True
    accent_color: tuple = (79, 142, 247)   # blue
    dark_color: tuple   = (7, 8, 15)        # bg
    light_color: tuple  = (221, 225, 245)   # text


@dataclass
class ReportSection:
    title: str
    content: str
    chart_image: Optional[bytes] = None
    table_data: Optional[pd.DataFrame] = None


@dataclass
class ReportData:
    config: ReportConfig
    profile: DatasetProfile
    df: pd.DataFrame
    sections: List[ReportSection]
    generated_at: str = ""

    def __post_init__(self):
        self.generated_at = datetime.now().strftime("%B %d, %Y — %H:%M")
