#!/usr/bin/env bash
# Simple runner: reads config.json and runs main.py with --use-apis
CONFIG=config.json
if [ ! -f "$CONFIG" ]; then
  echo "config.json not found!"
  exit 1
fi
TOKEN=$(python - <<PY
import json
c=json.load(open('config.json'))
print(c.get('aqicn_token',''))
PY
)
STATION=$(python - <<PY
import json
c=json.load(open('config.json'))
print(c.get('aqicn_station',''))
PY
)
CETESB_LAYER=$(python - <<PY
import json
c=json.load(open('config.json'))
print(c.get('cetesb_layer',''))
PY
)
IEMA_CSV=$(python - <<PY
import json
c=json.load(open('config.json'))
print(c.get('iema_csv',''))
PY
)
echo "Executando com token AQICN: ${TOKEN:0:6}... (oculto)"
python main.py --use-apis --aqicn-token "$TOKEN" --aqicn-station "$STATION" --cetesb-layer "$CETESB_LAYER" --iema-csv "$IEMA_CSV"
