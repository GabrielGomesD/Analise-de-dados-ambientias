\
@echo off
REM Simple runner for Windows: reads config.json and runs main.py with --use-apis
for /f "delims=" %%A in ('python - <<PY
import json
c=json.load(open('config.json'))
print(c.get('aqicn_token',''))
PY') do set TOKEN=%%A
for /f "delims=" %%A in ('python - <<PY
import json
c=json.load(open('config.json'))
print(c.get('aqicn_station',''))
PY') do set STATION=%%A
for /f "delims=" %%A in ('python - <<PY
import json
c=json.load(open('config.json'))
print(c.get('cetesb_layer',''))
PY') do set CETESB=%%A
for /f "delims=" %%A in ('python - <<PY
import json
c=json.load(open('config.json'))
print(c.get('iema_csv',''))
PY') do set IEMA=%%A
echo Executando com token AQICN: %TOKEN:~0,6%... (oculto)
python main.py --use-apis --aqicn-token %TOKEN% --aqicn-station %STATION% --cetesb-layer %CETESB% --iema-csv "%IEMA%"
