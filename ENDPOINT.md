# Inference endpoint

| | |
|---|---|
| **IP** | `10.17.0.156` (VM2, on the DEI network or DEI-VPN) |
| **Port** | `8000` |
| **Classification** | `POST http://10.17.0.156:8000/classify` |
| **Health check** | `GET http://10.17.0.156:8000/health` |
| **Interactive docs** | `http://10.17.0.156:8000/docs` |

No authentication is needed.

## Request and response

Send a batch of flights with the raw BTS column names. Other columns are ignored.
The answer has one label per flight, in the same order: `on_time`, `moderate_delay`
or `severe_delay`.

```bash
curl -X POST http://10.17.0.156:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"instances": [{"FL_DATE": "2023-07-02", "OP_UNIQUE_CARRIER": "9E",
       "OP_CARRIER_FL_NUM": 4900, "TAIL_NUM": "N307PQ", "ORIGIN": "DTW",
       "ORIGIN_STATE_ABR": "MI", "DEST": "DSM", "DEST_STATE_ABR": "IA",
       "CRS_DEP_TIME": 900, "DEP_TIME_BLK": "0900-0959", "CRS_ARR_TIME": 1042,
       "ARR_TIME_BLK": "1000-1059", "CRS_ELAPSED_TIME": 102.0, "DISTANCE": 528.0}]}'
```

```json
{"classifications": ["on_time"]}
```

`/classify` only accepts POST, so opening it in a browser gives `405 Method Not Allowed`.
To try it by hand, use `/docs` → `POST /classify` → **Try it out**.

Invalid requests get a `422` that says which flight and field is wrong. More details
are in [api/README.md](api/README.md).
