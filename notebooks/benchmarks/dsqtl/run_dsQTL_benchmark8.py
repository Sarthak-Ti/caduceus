"""Execute the self-contained notebook and checkpoint outputs after each cell."""
from pathlib import Path
import json, os, time, traceback, sys
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent
PATH=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'dsQTL_benchmark8.ipynb'
ROOT=PATH.parent
CACHE=ROOT/'dsQTL_benchmark8_cache'
CACHE.mkdir(exist_ok=True)
nb=nbformat.read(PATH,as_version=4)

def status(state,**kwargs):
    value=dict(state=state,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               slurm_job_id=os.environ.get('SLURM_JOB_ID'),**kwargs)
    temp=CACHE/'run_status.tmp.json';temp.write_text(json.dumps(value,indent=2));temp.replace(CACHE/'run_status.json')
    print(json.dumps(value),flush=True)

def save():
    temp=PATH.with_suffix('.tmp.ipynb');nbformat.write(nb,temp);temp.replace(PATH)

def before(cell,cell_index,**kwargs):
    status('running',cell=cell_index)

def after(cell,cell_index,**kwargs):
    save();status('running',completed_cell=cell_index)

try:
    status('starting')
    client=NotebookClient(nb,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},
                          on_cell_execute=before,on_cell_executed=after)
    client.execute()
    save();status('complete',notebook=str(PATH))
except BaseException:
    save();status('failed',traceback=traceback.format_exc())
    raise
