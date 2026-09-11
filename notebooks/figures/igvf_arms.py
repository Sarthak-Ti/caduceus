"""Canonical arm registry for the IGVF meeting figures.

One place defines, for every model arm that appears in any panel:
  * `id`     - the stable key the cache is written under
  * `label`  - the display name, identical in every figure
  * `family` - which model family it belongs to
  * `color`  - its hue, a shade of the family colour
  * `grids`  - which benchmarks the arm can actually appear in

`grids` is the field that encodes real constraints rather than taste. The
no-pretraining arm predicts expression only, so it exists on the Enformer CAGE
grid and in the eQTL benchmark but NOT in dsQTL; listing that here means a panel
cannot silently include an arm that has no scores, and `arms_for()` is the only
thing a figure needs to call.

Renaming an arm across every figure is a one-line edit to `label`. Recolouring
is a one-line edit to `color`. Neither requires touching the pipeline or
rebuilding the cache: the cache is keyed on `id`, which never changes.

Colour convention (set by the presenter, 2026-09-09):
    cis-Mamba   green      Borzoi      blue
    Enformer    red        AlphaGenome gold
    ChromBPNet  purple     references  grey
Arms inside one family are shades of that family's hue, darkest = the base or
headline model, lighter = further from it. Identity never rests on colour alone:
every bar carries an axis label and a printed value.
"""
from dataclasses import dataclass, field

# --- family base hues --------------------------------------------------------
C_OURS = '#177A3B'   # cis-Mamba (this work)
C_ENF = '#A32316'    # Enformer family
C_BOR = '#12456F'    # Borzoi family
C_AG = '#C9930A'     # AlphaGenome
C_CBP = '#6A3D9A'    # ChromBPNet
C_REF = '#6B7280'    # measurements / nulls / chance -- never a model

FAMILY = {'ours': C_OURS, 'enformer': C_ENF, 'borzoi': C_BOR,
          'alphagenome': C_AG, 'chrombpnet': C_CBP, 'reference': C_REF}

# Which grid/benchmark each arm can appear in. Named so a panel reads as English.
ENFORMER_GRID = 'enformer_grid'   # 128 bp GM12878 CAGE, Enformer test regions
BORZOI_GRID = 'borzoi_grid'       # 32 bp GM12878 CAGE, Borzoi fold3, strand-separate
EQTL = 'eqtl'                     # LCL eQTL variant effect
DSQTL = 'dsqtl'                   # GM12878 dsQTL variant effect


@dataclass(frozen=True)
class Arm:
    id: str
    label: str
    family: str
    color: str
    grids: frozenset = field(default_factory=frozenset)
    hidden: bool = False        # scored and cached, but drawn by no panel


def _a(id, label, family, color, *grids, hidden=False):
    return Arm(id, label, family, color, frozenset(grids), hidden)


# --- the registry ------------------------------------------------------------
# Order here is the canonical order arms are listed in when a panel does not
# sort them by value.
ARMS = [
    # cis-Mamba. The Borzoi-grid arm and the Enformer-grid headline arm deliberately
    # share both the darkest green and the label: they are the same method, and they
    # never appear in the same panel, so "our model" reads as one thing throughout.
    _a('mamba_1ct', 'cis-Mamba (1 cell type)', 'ours', '#177A3B',
       ENFORMER_GRID, EQTL, DSQTL),
    _a('mamba_nopre', 'cis-Mamba (no pretraining)', 'ours', '#63BF82',
       ENFORMER_GRID, EQTL),                       # expression only -- no dsQTL arm exists
    _a('mamba_8ct', 'cis-Mamba (8 cell types)', 'ours', '#9BD9AE',
       ENFORMER_GRID, EQTL),
    # The Borzoi-grid Caduceus run. It is a different checkpoint from the
    # Enformer-grid arm above -- different grid, different training run -- so it
    # keeps its own id and its own cache key, but it carries the same label and
    # the same green, because it is the same method and the two never appear in
    # one panel. Panel 2b overrides the label locally to name the training data;
    # see `panel2b_accuracy_borzoi`. The epoch-10 run is no longer shown anywhere.
    _a('mamba_1ct_borzoi', 'cis-Mamba (1 cell type)', 'ours', '#177A3B', BORZOI_GRID),

    # Enformer family
    _a('enformer_pre', 'Enformer (pretrained)', 'enformer', '#A32316',
       ENFORMER_GRID, EQTL, DSQTL),
    # Two 1-cell-type retrained Enformers were run, at 768 and 1536 channels. Only
    # the 1536ch one is shown, and it is labelled plainly as the 1-CT retrain -- the
    # width is an implementation detail the audience does not need. The 768ch run
    # stays registered and stays in the cache (its ladder and boxplot numbers are
    # still built and gated); `hidden` only removes it from the panels.
    _a('enformer_n1', 'Enformer (retrained, 1 CT, 768ch)', 'enformer', '#E2603F',
       ENFORMER_GRID, hidden=True),
    _a('enformer_n1_1536', 'Enformer (retrained, 1 CT)', 'enformer', '#C93B22',
       ENFORMER_GRID),
    _a('enformer_n16', 'Enformer (retrained, 16 CTs)', 'enformer', '#F0907A',
       ENFORMER_GRID),
    _a('enformer_n42', 'Enformer (retrained, 42 CTs)', 'enformer', '#F8BFB2',
       ENFORMER_GRID),
    _a('enformer_retrained_dsqtl', 'Enformer (retrained, GM12878)', 'enformer', '#C93B22',
       DSQTL),

    # Borzoi family
    _a('borzoi_pre', 'Borzoi (pretrained)', 'borzoi', '#12456F',
       BORZOI_GRID, EQTL, DSQTL),
    _a('borzoi_lora', 'Borzoi (LoRA)', 'borzoi', '#7FB3DE', BORZOI_GRID, EQTL, DSQTL),
    _a('borzoi_trunk', 'Borzoi (full FT)', 'borzoi', '#2E7EBD', BORZOI_GRID, EQTL, DSQTL),
    _a('borzoi_scratch', 'Borzoi (from scratch)', 'borzoi', '#1B639C',
       BORZOI_GRID, EQTL, DSQTL),

    # AlphaGenome. Not held out on the Borzoi fold3 grid -- panels that show it
    # there must say so; see NOT_HELD_OUT below.
    _a('alphagenome', 'AlphaGenome', 'alphagenome', '#C9930A',
       ENFORMER_GRID, BORZOI_GRID, EQTL, DSQTL),

    # ChromBPNet, dsQTL only (two experiments x two scorers)
    _a('cbp_dnase_logfc', 'ChromBPNet (DNase logFC)', 'chrombpnet', '#6A3D9A', DSQTL),
    _a('cbp_dnase_ips', 'ChromBPNet (DNase IPS)', 'chrombpnet', '#9B7BC4', DSQTL),
    _a('cbp_atac_logfc', 'ChromBPNet (ATAC logFC)', 'chrombpnet', '#4A2870', DSQTL),
    _a('cbp_atac_ips', 'ChromBPNet (ATAC IPS)', 'chrombpnet', '#C0AEDC', DSQTL),

    # References. Neither is a model; both are read against the model rows.
    _a('null_avg', '[null] cross-CT average', 'reference', C_REF,
       ENFORMER_GRID, BORZOI_GRID),
    _a('acc_atac', 'GM12878 ATAC (measured)', 'reference', C_REF, ENFORMER_GRID),
    _a('acc_dnase', 'GM12878 DNase (measured)', 'reference', C_REF, BORZOI_GRID),
]

BY_ID = {a.id: a for a in ARMS}

# Arms whose training data is not held out from the grid they are shown on.
# AlphaGenome's training set is not public and its fold3 status is unknown, so
# the Borzoi-grid panels annotate it rather than dropping it.
NOT_HELD_OUT = {(BORZOI_GRID, 'alphagenome'): 'not held out'}


# --- source-notebook aliases -------------------------------------------------
# Every notebook this refactor draws from named the same checkpoint differently.
# These maps are the whole reason the naming was inconsistent; they exist so the
# pipeline can translate once, at read time, and nothing downstream ever sees a
# source-specific label again.
ALIASES = {
    # comparing_model6 §1 E_SPEC labels -> arm id
    ENFORMER_GRID: {
        'cis-Mamba (1 cell type)': 'mamba_1ct',
        'cis-Mamba, no pretraining': 'mamba_nopre',
        'cis-Mamba (8 cell types)': 'mamba_8ct',
        'Enformer (pretrained)': 'enformer_pre',
        'AlphaGenome': 'alphagenome',
        'Enformer retrained, 1 CT': 'enformer_n1',
        'Enformer retrained, 1 CT (1536ch)': 'enformer_n1_1536',
        'Enformer retrained, 16 CTs': 'enformer_n16',
        'Enformer retrained, 42 CTs': 'enformer_n42',
        '[null] cross-CT average': 'null_avg',
        'GM12878 ATAC (measured)': 'acc_atac',
    },
    # comparing_model5 §1 / comparing_model6 §2 keys -> arm id
    BORZOI_GRID: {
        'borzoi_pre': 'borzoi_pre',
        'borzoi_lora': 'borzoi_lora',
        'borzoi_trunk': 'borzoi_trunk',
        'borzoi_scratch': 'borzoi_scratch',
        'cad_ep4': 'mamba_1ct_borzoi',
        'alphagenome': 'alphagenome',
        '__avg__': 'null_avg',
        '__acc__': 'acc_dnase',
    },
    # IGVF_meeting_figures load_eqtl() labels -> arm id
    EQTL: {
        'cis-Mamba (1 cell type)': 'mamba_1ct',
        'cis-Mamba, no pretraining': 'mamba_nopre',
        'cis-Mamba (8 cell types)': 'mamba_8ct',
        'Enformer (pretrained)': 'enformer_pre',
        'Borzoi (pretrained)': 'borzoi_pre',
        'Borzoi LoRA (GM12878)': 'borzoi_lora',
        'Borzoi full FT (GM12878)': 'borzoi_trunk',
        'Borzoi from scratch (GM12878)': 'borzoi_scratch',
        'AlphaGenome': 'alphagenome',
    },
    # dsQTL_benchmark8_analysis ARMS labels -> arm id
    DSQTL: {
        'cis-Mamba (GM12878 DNase logFC)': 'mamba_1ct',
        'Enformer (pretrained)': 'enformer_pre',
        'Enformer retrained (GM12878)': 'enformer_retrained_dsqtl',
        'Borzoi (pretrained)': 'borzoi_pre',
        'Borzoi LoRA (GM12878)': 'borzoi_lora',
        'Borzoi full FT (GM12878)': 'borzoi_trunk',
        'Borzoi from scratch (GM12878)': 'borzoi_scratch',
        'AlphaGenome': 'alphagenome',
        'ChromBPNet (GM12878 DNase logFC)': 'cbp_dnase_logfc',
        'ChromBPNet (GM12878 DNase IPS)': 'cbp_dnase_ips',
        'ChromBPNet (GM12878 ATAC logFC)': 'cbp_atac_logfc',
        'ChromBPNet (GM12878 ATAC IPS)': 'cbp_atac_ips',
    },
}


def to_id(grid, source_label):
    """Translate a source-notebook label/key into a canonical arm id."""
    m = ALIASES[grid]
    assert source_label in m, f'{grid}: no arm id registered for {source_label!r}'
    return m[source_label]


def arms_for(grid, exclude=(), families=None, include_reference=True,
             include_hidden=False):
    """The arms valid on `grid`, in registry order.

    `exclude` takes arm ids. `families` restricts to a set of family names.
    Reference rows (nulls, measured assays) are dropped by `include_reference=False`.
    Arms flagged `hidden` are omitted unless `include_hidden=True`.
    """
    out = []
    for a in ARMS:
        if grid not in a.grids or a.id in exclude:
            continue
        if a.hidden and not include_hidden:
            continue
        if families is not None and a.family not in families:
            continue
        if not include_reference and a.family == 'reference':
            continue
        out.append(a)
    return out


def label(arm_id, grid=None):
    """Display label, with the held-out caveat appended where it applies."""
    a = BY_ID[arm_id]
    note = NOT_HELD_OUT.get((grid, arm_id)) if grid else None
    return f'{a.label} ({note})' if note else a.label


def color(arm_id):
    return BY_ID[arm_id].color
