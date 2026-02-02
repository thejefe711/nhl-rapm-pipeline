import hashlib
import pandas as pd
from typing import Any, List

class IncrementalValidator:
    def hash_result(self, result: Any) -> str:
        """Create a hash of the result for comparison"""
        if isinstance(result, pd.DataFrame):
            # Sort by index or columns to ensure stability
            return hashlib.md5(pd.util.hash_pandas_object(result, index=True).values).hexdigest()
        return hashlib.md5(str(result).encode()).hexdigest()

    def validate_idempotency(self, pipeline, input_data) -> bool:
        """Run pipeline twice, verify identical output"""
        # Note: This assumes 'pipeline' has a 'run' method and is stateless or reset-able
        result1 = pipeline.run(input_data)
        result2 = pipeline.run(input_data)
        return self.hash_result(result1) == self.hash_result(result2)
    
    def validate_incremental_vs_full(self, pipeline, full_data, incremental_chunks) -> bool:
        """Compare incremental processing to full rebuild"""
        # Note: This assumes 'pipeline' has 'run_full' and 'run_incremental' methods
        full_result = pipeline.run_full(full_data)
        
        incremental_result = None
        for chunk in incremental_chunks:
            incremental_result = pipeline.run_incremental(chunk, incremental_result)
        
        return self.compare_results(full_result, incremental_result, tolerance=0.001)

    def compare_results(self, res1, res2, tolerance=0.001) -> bool:
        if isinstance(res1, pd.DataFrame) and isinstance(res2, pd.DataFrame):
            try:
                # Align indices and columns
                # This is a simplified comparison
                pd.testing.assert_frame_equal(res1, res2, atol=tolerance)
                return True
            except AssertionError:
                return False
        return res1 == res2
