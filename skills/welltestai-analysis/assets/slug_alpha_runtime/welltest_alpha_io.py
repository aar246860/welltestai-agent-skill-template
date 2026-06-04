from __future__ import annotations

from pathlib import Path

import pandas as pd

from welltest_alpha_schema import CaseMetadata, RawAlphaCase


def load_case(
    csv_path: str | Path,
    *,
    case_id: str | None = None,
    test_type: str,
    metadata: CaseMetadata,
) -> RawAlphaCase:
    path = Path(csv_path)
    data = pd.read_csv(path)
    metadata.source_path = str(path)
    return RawAlphaCase(case_id=case_id or path.stem, test_type=test_type, data=data, metadata=metadata)
