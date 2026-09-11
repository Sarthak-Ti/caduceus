"""Build derived EP10 caches and/or execute its notebook; no model inference or DNase rescan."""
from pathlib import Path
import argparse,ast,json,os,time,traceback
import ep_linking10_pipeline as p

NOTEBOOK=Path(__file__).with_name('EP_linking10.ipynb')


def status(state,stage,**kwargs):
    p.CACHE.mkdir(parents=True,exist_ok=True)
    d=dict(state=state,stage=stage,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs)
    p.atomic_json(p.CACHE/'run_status.json',d);print(json.dumps(d),flush=True)


def validate_notebook():
    import nbformat
    nb=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(nb)
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code':ast.parse(c.source,filename=f'EP10:cell{i}')
    return nb


def execute_notebook(stage):
    import nbformat
    from nbclient import NotebookClient
    nb=validate_notebook()
    def save():
        temp=NOTEBOOK.with_suffix('.tmp.ipynb');nbformat.write(nb,temp);temp.replace(NOTEBOOK)
    def before(cell,cell_index,**kwargs):status('running',stage,cell=cell_index)
    def after(cell,cell_index,**kwargs):save();status('running',stage,completed_cell=cell_index)
    client=NotebookClient(nb,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(NOTEBOOK.parent.resolve())}},
                          on_cell_execute=before,on_cell_executed=after)
    try:client.execute()
    finally:save()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['build','notebook','all'],nargs='?',default='all')
    args=parser.parse_args(argv)
    try:
        status('starting',args.stage)
        if args.stage in ['notebook','all']:validate_notebook()
        if args.stage in ['build','all']:p.build_cache()
        if args.stage in ['notebook','all']:execute_notebook(args.stage)
        status('complete',args.stage,notebook=str(NOTEBOOK.resolve()))
    except BaseException:
        status('failed',args.stage,traceback=traceback.format_exc());raise


if __name__=='__main__':main()
