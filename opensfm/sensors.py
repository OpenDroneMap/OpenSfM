# pyre-strict
from functools import lru_cache
from typing import Any, Dict, List

import yaml
import os
import sqlite3

from opensfm import context, io

@lru_cache(1)
def sensor_data() -> Dict[str, Any]:
    if os.path.isfile(context.SENSOR_DATA_DB):
        try:
            conn = sqlite3.connect(context.SENSOR_DATA_DB)
            cur = conn.cursor()
            cur.execute("SELECT id,focal FROM sensors")
            rows = cur.fetchall()
            conn.close()
            return {r[0]: r[1] for r in rows}
        except Exception as e:
            print("Cannot query %s: %s" % (context.SENSOR_DATA_DB, str(e)))
            # Fallback to sensor_data.json

    with io.open_rt(context.SENSOR_DATA) as f:
        data = io.json_load(f)

    # Convert model types to lower cases for easier query
    return {k.lower(): v for k, v in data.items()}


@lru_cache(1)
def camera_calibration() -> List[Dict[str, Any]]:
    with io.open_rt(context.CAMERA_CALIBRATION) as f:
        data = yaml.safe_load(f)
    return data
