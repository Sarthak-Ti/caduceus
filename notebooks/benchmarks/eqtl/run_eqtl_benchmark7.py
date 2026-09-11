"""Staged cache/notebook execution; validate and reuse completed extraction products."""
from pathlib import Path
import argparse, ast, json, os, sys, time, traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eqtl_benchmark7_pipeline import CACHE, build_expression_cache
import eqtl_benchmark7_analysis as analysis

NOTEBOOK = ROOT / 'eQTL_benchmark7.ipynb'


def status(state, stage, **kwargs):
    CACHE.mkdir(exist_ok=True)
    value = dict(state=state, stage=stage,
                 updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 slurm_job_id=os.environ.get('SLURM_JOB_ID'), **kwargs)
    temp = CACHE / 'run_status.tmp.json'
    temp.write_text(json.dumps(value, indent=2)); temp.replace(CACHE / 'run_status.json')
    print(json.dumps(value), flush=True)


def validate_notebook(path=NOTEBOOK):
    import nbformat
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            ast.parse(cell.source, filename=f'{path.name}:cell-{i}')
    return nb


def run_notebook(stage='notebook'):
    import nbformat
    from nbclient import NotebookClient
    nb = validate_notebook()

    def save():
        temp = NOTEBOOK.with_suffix('.tmp.ipynb')
        nbformat.write(nb, temp); temp.replace(NOTEBOOK)

    def before(cell, cell_index, **kwargs):
        status('running', stage, cell=cell_index)

    def after(cell, cell_index, **kwargs):
        save(); status('running', stage, completed_cell=cell_index)

    client = NotebookClient(nb, timeout=None, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}},
                            on_cell_execute=before, on_cell_executed=after)
    try:
        client.execute()
    finally:
        # Preserve traceback outputs if a cell fails as well as successful cells.
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['cache', 'notebook', 'all'], nargs='?', default='all')
    parser.add_argument('--validate-score-sources', action='store_true',
                        help='Check original prediction provenance; default uses the score cache.')
    args = parser.parse_args(argv)
    try:
        status('starting', args.stage)
        if args.stage in ('notebook', 'all'):
            validate_notebook()  # Fail before any costly work if cells are malformed.
        analysis.score_pairs(validate_sources=args.validate_score_sources)
        if args.stage in ('cache', 'all'):
            status('validating/building expression cache', args.stage)
            build_expression_cache()
        if args.stage in ('notebook', 'all'):
            status('executing notebook', args.stage)
            run_notebook(args.stage)
        status('complete', args.stage, notebook=str(NOTEBOOK))
    except BaseException:
        status('failed', args.stage, traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
