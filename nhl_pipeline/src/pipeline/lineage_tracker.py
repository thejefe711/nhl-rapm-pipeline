import duckdb
import time
import json
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Callable

@dataclass
class LineageRecord:
    stage: str
    timestamp: datetime
    input_hash: Optional[str]
    output_hash: Optional[str]
    row_count_in: Optional[int]
    row_count_out: Optional[int]
    transformation: str
    parameters: dict
    duration_seconds: float

class LineageContext:
    def __init__(self, set_input: Callable, set_output: Callable):
        self._set_input = set_input
        self._set_output = set_output
    
    def set_input(self, hash_val: str, count: int):
        self._set_input(hash_val, count)
        
    def set_output(self, hash_val: str, count: int):
        self._set_output(hash_val, count)

class LineageTracker:
    def __init__(self, db_path: str):
        self.db = duckdb.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS lineage_log (
                stage VARCHAR,
                timestamp TIMESTAMP,
                input_hash VARCHAR,
                output_hash VARCHAR,
                row_count_in INTEGER,
                row_count_out INTEGER,
                transformation VARCHAR,
                parameters JSON,
                duration_seconds DOUBLE
            )
        """)
    
    def _record(self, stage, input_hash, output_hash, row_count_in, row_count_out, params, duration):
        self.db.execute("""
            INSERT INTO lineage_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            stage, datetime.now(), input_hash, output_hash, 
            row_count_in, row_count_out, "standard_transform", 
            json.dumps(params), duration
        ])

    @contextmanager
    def track_stage(self, stage_name: str, params: dict = {}):
        start = time.time()
        input_info = {'hash': None, 'count': None}
        output_info = {'hash': None, 'count': None}
        
        def set_input(h, c):
            input_info['hash'] = h
            input_info['count'] = c
            
        def set_output(h, c):
            output_info['hash'] = h
            output_info['count'] = c
        
        try:
            yield LineageContext(set_input, set_output)
        finally:
            duration = time.time() - start
            self._record(
                stage_name, 
                input_info['hash'], output_info['hash'], 
                input_info['count'], output_info['count'], 
                params, duration
            )
