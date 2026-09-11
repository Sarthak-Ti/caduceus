"""Analysis of K562 enhancer breadth; shared cohorts and paired component bootstrap."""
from ep_linking10_pipeline import *
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS=['supported','eligible','higher','lower','inactive','missing']
BOOT_GROUPS=['supported','eligible','higher','lower']
MODEL_COLORS=dict(zip(MODELS,['#2878B5','#69A6D2','#164F7A','#218C74','#DD8452','#7853A6']))
RANK_CONTRASTS=[(('ours-borzoi','shuffle'),('Borzoi','shuffle')),
                (('ours-enformer','shuffle'),('Enformer','shuffle')),
                (('ours-striped','shuffle'),('Borzoi','shuffle'))]
for _model in ['ours-borzoi','ours-enformer','ours-striped']:
    RANK_CONTRASTS.extend([((_model,'acc'),(_model,'shuffle')),
                           ((_model,'shuffle+acc'),(_model,'shuffle'))])


def dependency_components(q):
    """Connect pairs sharing a target gene or overlapping enhancer intervals.

    Construct on all mapped pairs before filtering. Connected components can cross
    specificity strata; resample them jointly. Does not model arbitrary distal LD.
    """
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y:parent[y]=x
    for r in q.itertuples():union('e:'+r.enhancer_id,'g:'+r.gene_id)
    enh=q[['enhancer_id','chrom','enhancer_start','enhancer_end']].drop_duplicates('enhancer_id')
    for _,group in enh.groupby('chrom'):
        end=-1; representative=None
        for r in group.sort_values(['enhancer_start','enhancer_end']).itertuples():
            if r.enhancer_start<0 or r.enhancer_end<=r.enhancer_start:continue
            node='e:'+r.enhancer_id
            if representative is not None and r.enhancer_start<end:
                union(representative,node);end=max(end,r.enhancer_end)
            else:representative=node;end=r.enhancer_end
    roots=[find('e:'+e) for e in q.enhancer_id]
    return pd.factorize(np.asarray(roots),sort=True)[0]


def cohort_masks(data):
    ctx=data['context']; valid=data['valid']
    long=np.array([MODELS[m]['reach']>=262144 for m,_,_ in RUNS])
    allcols=np.arange(len(RUNS)); longcols=np.flatnonzero(long)
    reference=next(j for j,r in enumerate(RUNS) if r[:2]==('ours-borzoi','acc'))
    base=ctx[:,reference]
    assert np.all(ctx[base][:,longcols]), 'A claimed long-context model does not support the reference subset'
    return {'common':dict(context=ctx.all(1),support=(ctx & valid).all(1),columns=allcols),
            'reach262':dict(context=base,support=base & valid[:,longcols].all(1),columns=longcols)}


def subset_mask(q,support,group):
    if group=='supported':return support.copy()
    if group=='eligible':return support & q.k562_active.to_numpy(dtype=bool)
    return support & q.specificity.eq(group).to_numpy()


def ap_with_weights(y, score, weights):
    """Tie-correct weighted AP; used for validation and arbitrary weights."""
    order=np.argsort(-score,kind='stable');y=np.asarray(y)[order];score=np.asarray(score)[order]
    ends=np.r_[np.flatnonzero(np.diff(score)),len(score)-1] if len(score) else np.array([],int)
    return sorted_ap(y,np.asarray(weights)[order],ends)


def sorted_ap(y,weights,ends):
    total_pos=np.dot(y,weights);total=weights.sum()
    if total_pos<=0 or total_pos>=total:return np.nan
    tp=np.cumsum(y*weights)[ends]; denom=np.cumsum(weights)[ends]
    precision=np.divide(tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0)
    return float(np.dot(np.diff(np.r_[0.,tp])/total_pos,precision))


def summary_row(q,mask):
    x=q.loc[mask];pos=x[x.label.eq(1)]
    return dict(n=len(x),positives=len(pos),negatives=len(x)-len(pos),prevalence=len(pos)/len(x) if len(x) else np.nan,
        n_enhancers=x.enhancer_id.nunique(),n_genes=x.gene_id.nunique(),n_components=x.component.nunique(),
        positive_enhancers=pos.enhancer_id.nunique(),positive_genes=pos.gene_id.nunique(),positive_components=pos.component.nunique(),
        median_distance=x.distance_to_TSS.median(),median_length=x.enhancer_length.median(),
        median_k562_signal=x.k562_signal_mean.median(),median_breadth=x.breadth.median())


def point_metrics(y,s):
    if np.unique(y).size<2:return np.nan,np.nan
    return average_precision_score(y,s),roc_auc_score(y,s)


def reproduce_benchmark(q,data):
    """Reproduce EP_linking9 AP (including its explicit out-of-context zeros)."""
    cohorts=cohort_masks(data);rows=[]
    for j,(model,arm,_) in enumerate(RUNS):
        for name,mask in [('own_context',data['context'][:,j]),('all_rows',np.ones(len(q),bool)),
                          ('common_context',cohorts['common']['context']),
                          ('reach262_context',cohorts['reach262']['context'])]:
            if name=='reach262_context' and j not in cohorts['reach262']['columns']:continue
            ap,auc=point_metrics(q.label.to_numpy()[mask],np.abs(data['signed'][mask,j]))
            rows.append(dict(model=model,arm=arm,cohort=name,n=int(mask.sum()),positives=int(q.label.to_numpy()[mask].sum()),AP=ap,AUROC=auc))
    out=pd.DataFrame(rows);out.to_csv(CACHE/'benchmark_reproduction.tsv',sep='\t',index=False);return out


def evaluate(q,data,n_boot=400,seed=42):
    """Joint component weights across strata AND arms; preserve cross-stratum dependence."""
    q=q.copy();q['component']=dependency_components(q)
    scores=np.abs(data['signed']);y=q.label.to_numpy();rng=np.random.default_rng(seed)
    metrics=[]; attrition=[]; draw_cache={}; distrows=[]
    attrition.append(dict(cohort='mapped',stratum='all',**summary_row(q,np.ones(len(q),bool))))
    for cohort,cfg in cohort_masks(data).items():
        support=cfg['support'];cols=cfg['columns']
        attrition.append(dict(cohort=cohort,stratum='context_before_validity',**summary_row(q,cfg['context'])))
        _,inv=np.unique(q.component.to_numpy()[support],return_inverse=True)
        ncomp=len(np.unique(inv)); idxsupport=np.flatnonzero(support)
        weights=np.zeros((n_boot,len(q)),np.uint16)
        for b in range(n_boot):
            mult=rng.multinomial(ncomp,np.full(ncomp,1/ncomp)) if ncomp else np.array([],int)
            weights[b,idxsupport]=mult[inv]
        for group in GROUPS:
            mask=subset_mask(q,support,group);idx=np.flatnonzero(mask);counts=summary_row(q,mask)
            attrition.append(dict(cohort=cohort,stratum=group,**counts))
            boot=np.full((n_boot,len(RUNS)),np.nan)
            for j in cols:
                ap,auc=point_metrics(y[idx],scores[idx,j])
                if group in BOOT_GROUPS and counts['n_components']>=2 and np.unique(y[idx]).size==2:
                    order=np.argsort(-scores[idx,j],kind='stable');ordered=idx[order]
                    ends=np.r_[np.flatnonzero(np.diff(scores[ordered,j])),len(ordered)-1]
                    for b in range(n_boot):boot[b,j]=sorted_ap(y[ordered],weights[b,ordered],ends)
                good=boot[:,j][np.isfinite(boot[:,j])]
                lo,hi=np.percentile(good,[2.5,97.5]) if len(good) else (np.nan,np.nan)
                metrics.append(dict(cohort=cohort,stratum=group,model=RUNS[j][0],arm=RUNS[j][1],
                                    AP=ap,AUROC=auc,AP_ci_low=lo,AP_ci_high=hi,valid_bootstraps=len(good),**counts))
            draw_cache[(cohort,group)]=boot
            # Distance diagnostics are descriptive; avoid claiming power in tiny subgroups.
            if group in ['higher','lower']:
                for label,dmask in [('<=10kb',q.distance_to_TSS.le(10000)),('10-100kb',q.distance_to_TSS.gt(10000)&q.distance_to_TSS.le(100000)),('>100kb',q.distance_to_TSS.gt(100000))]:
                    take=mask & dmask.to_numpy();dy=y[take]
                    for j in cols:
                        ap,auc=point_metrics(dy,scores[take,j])
                        distrows.append(dict(cohort=cohort,stratum=group,distance_bin=label,model=RUNS[j][0],arm=RUNS[j][1],AP=ap,AUROC=auc,**summary_row(q,take)))
        progress('Evaluated',cohort,'with',int(support.sum()),'pairs and',ncomp,'dependency components')
    table=pd.DataFrame(metrics); counts=pd.DataFrame(attrition)
    table.to_csv(CACHE/'stratified_metrics.tsv',sep='\t',index=False)
    counts.to_csv(CACHE/'cohort_counts.tsv',sep='\t',index=False)
    pd.DataFrame(distrows).to_csv(CACHE/'distance_diagnostics.tsv',sep='\t',index=False)
    np.savez_compressed(CACHE/'bootstrap_draws.npz',**{f'{a}__{b}':v for (a,b),v in draw_cache.items()})
    comps=q.groupby('component').agg(n_pairs=('pair_id','size'),n_genes=('gene_id','nunique'),n_enhancers=('enhancer_id','nunique'),positives=('label','sum'))
    comps.to_csv(CACHE/'dependency_components.tsv',sep='\t')
    for j,name in enumerate(NAMES):
        q[name+' signed']=data['signed'][:,j];q[name+' context']=data['context'][:,j]
        q[name+' valid']=data['valid'][:,j];q[name+' shuffled']=data['shuffled'][:,j]
    q.to_csv(CACHE/'pair_scores_and_specificity.tsv.gz',sep='\t',index=False)
    atomic_json(CACHE/'analysis_manifest.json',dict(version=1,n_boot=n_boot,seed=seed,
        clusters='connected components of shared target genes and overlapping enhancer intervals, constructed on all mapped pairs',
        resampling='multinomial component weights jointly across specificity groups and model arms within each cohort',
        limitations='No arbitrary distal LD or study-level dependence modeled; large components reduce effective sample size; intervals descriptive',
        code_sha256=digest(Path(__file__)),source_cache_sha256=digest(CACHE/'complete.json')))
    return q,table,counts,draw_cache


def paired_comparisons(table,draws):
    keys=[r[:2] for r in RUNS]
    contrasts=[(('ours-borzoi','shuffle'),('Borzoi','shuffle')),
               (('ours-enformer','shuffle'),('Enformer','shuffle')),
               (('ours-striped','shuffle'),('Borzoi','shuffle'))]
    for model in ['ours-borzoi','ours-enformer','ours-striped']:
        contrasts.extend([((model,'acc'),(model,'shuffle')),((model,'shuffle+acc'),(model,'shuffle'))])
    rows=[]
    for cohort,cfg in cohort_masks_from_table(table).items():
        for left,right in contrasts:
            a,b=keys.index(left),keys.index(right)
            if a not in cfg or b not in cfg:continue
            observed=[];samples=[]
            for group in ['higher','lower']:
                t=table[(table.cohort==cohort)&(table.stratum==group)].set_index(['model','arm'])
                observed.append(t.loc[left,'AP']-t.loc[right,'AP'])
                samples.append(draws[(cohort,group)][:,a]-draws[(cohort,group)][:,b])
            for label,obs,samp in zip(['higher','lower','higher_minus_lower'],[*observed,observed[0]-observed[1]],
                                     [*samples,samples[0]-samples[1]]):
                good=samp[np.isfinite(samp)];lo,hi=np.percentile(good,[2.5,97.5]) if len(good) else (np.nan,np.nan)
                rows.append(dict(cohort=cohort,left=' / '.join(left),right=' / '.join(right),contrast=label,
                    delta_AP=obs,ci_low=lo,ci_high=hi,valid_bootstraps=len(good),crosses_zero=lo<=0<=hi if len(good) else None))
    out=pd.DataFrame(rows);out.to_csv(CACHE/'paired_model_and_arm_differences.tsv',sep='\t',index=False);return out


def cohort_masks_from_table(table):
    return {c:[j for j,r in enumerate(RUNS) if ((t.model==r[0]) & (t.arm==r[1])).any()] for c,t in table.groupby('cohort')}


def sensitivity(enh):
    """Counts only, no AP-driven threshold choice; center readout and altered anchors secondary."""
    f=np.load(CACHE/'track_enhancer_fraction.npy',mmap_mode='r')
    center=np.load(CACHE/'track_enhancer_center_calls.npy',mmap_mode='r')
    rows=[]
    for name,frac,anchor,cut in [('primary',f,[121,122,123,625],.5),
                               ('presence_25pct',f,[121,122,123,625],.25),
                               ('presence_75pct',f,[121,122,123,625],.75),
                               ('center_bin',center,[121,122,123,625],.5),
                               ('including_altered_K562',f,[33,34,35,121,122,123,625],.5)]:
        classified,_=classify_enhancers(enh,frac,anchor,presence_fraction=cut)
        rows.append(dict(setting=name,median=classified.breadth_median.iloc[0],**classified.specificity.value_counts().to_dict()))
    out=pd.DataFrame(rows);out.to_csv(CACHE/'annotation_sensitivity.tsv',sep='\t',index=False);return out


def plot_results(table,cohort='common',arm='shuffle',diagnostic=False,xmin=.3):
    """One perturbation arm per figure; model colors agree across all views."""
    t=table[(table.cohort==cohort)&(table.arm==arm)&table.stratum.isin(['higher','lower'])]
    models=[m for m in MODELS if m in t.model.unique()]
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharex=True,sharey=True)
    vals=t.AP[np.isfinite(t.AP)]
    lo=0 if diagnostic or not len(vals) or vals.min()<xmin else xmin
    hi=min(1.,float(vals.max())+.12) if len(vals) and not diagnostic else 1.
    for ax,group in zip(axes,['lower','higher']):
        d=t[t.stratum==group].set_index('model').reindex(models)
        ax.barh(range(len(models)),d.AP-lo,left=lo,color=[MODEL_COLORS[m] for m in models],height=.62)
        for i,r in enumerate(d.itertuples()):
            if np.isfinite(r.AP):
                label_x=max(r.AP,r.AP_ci_high) if diagnostic and np.isfinite(r.AP_ci_high) else r.AP
                ax.text(min(label_x+.009,.965),i,f'{r.AP:.3f}',va='center',fontsize=9)
            if diagnostic and np.isfinite(r.AP_ci_low):ax.plot([r.AP_ci_low,r.AP_ci_high],[i,i],color='#333',lw=1.5)
        if diagnostic and np.isfinite(d.prevalence.iloc[0]):ax.axvline(d.prevalence.iloc[0],ls=':',color='#888')
        if not np.isfinite(d.AP).any():ax.text(.5,.5,'AP not estimable\nNeeds positive and negative pairs',transform=ax.transAxes,ha='center')
        r=d.iloc[0];ax.set_title(f'{group.capitalize()} specificity\nn={int(r.n)}, positives={int(r.positives)}, enhancers={int(r.n_enhancers)}')
        ax.set_xlim(lo,hi);ax.set_xlabel('AUPRC (average precision)');ax.grid(axis='x',alpha=.15)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(range(len(models)),models);axes[0].invert_yaxis()
    fig.suptitle(f'K562 enhancer–gene linking · {cohort} · {arm}')
    foot='95% component-bootstrap intervals; dotted line = prevalence.' if diagnostic else f'Both axes start at {lo:.2f}. Point estimates; uncertainty omitted.'
    fig.text(.5,.01,foot,ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.94])
    folder=CACHE/'figures';folder.mkdir(exist_ok=True)
    name=f'{cohort}_{arm.replace("+","_")}_{"diagnostic" if diagnostic else "presentation"}'
    for ext in ['png','pdf']:fig.savefig(folder/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    return fig


def plot_annotation_diagnostics(q,enh):
    """Feature-level accessibility/breadth and pair-level distance by cohort."""
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    active=enh[enh.k562_active]; x=np.log1p(active.k562_signal_mean);y=active.breadth
    axes[0].scatter(x,y,s=8,alpha=.3,color='#2878B5')
    rho=spearmanr(x,y).statistic if len(active)>2 else np.nan
    axes[0].set(xlabel='log(1 + K562 processed signal)',ylabel='Breadth (tracks)',title=f'Unique active enhancers; Spearman ρ={rho:.2f}')
    for group,color in [('higher','#2878B5'),('lower','#DD8452')]:
        e=active[active.specificity==group]
        axes[1].hist(np.log1p(e.enhancer_length),bins=30,alpha=.5,label=group,color=color)
    axes[1].set(xlabel='log(1 + enhancer length, bp)',ylabel='Unique enhancers',title='Length by specificity');axes[1].legend()
    for group,color in [('higher','#2878B5'),('lower','#DD8452')]:
        sub=q[q.specificity==group];axes[2].hist(np.log1p(sub.distance_to_TSS),bins=30,alpha=.5,label=group,color=color)
    axes[2].set(xlabel='log(1 + distance, bp)',ylabel='Mapped pairs',title='Distance before context filtering');axes[2].legend()
    fig.tight_layout();(CACHE/'figures').mkdir(exist_ok=True);fig.savefig(CACHE/'figures/annotation_diagnostics.png',dpi=180)
    return fig


def _macro_draws(unit_values, component_ids, component_weights):
    """Macro mean over gene-dataset units under shared component weights."""
    values=np.asarray(unit_values,float); ids=np.asarray(component_ids,int)
    out=np.full(len(component_weights),np.nan)
    for b,w in enumerate(component_weights):
        uw=w[ids]
        if uw.sum()>0: out[b]=np.average(values,weights=uw)
    return out


def within_gene_ranking(q,data,n_boot=1000,seed=73):
    """Macro AUROC within target gene, experimental dataset and specificity.

    Each informative unit contains at least one regulated and nonregulated pair.
    This removes cross-gene and cross-study comparisons from the ranking metric.
    Gene/overlap components are resampled jointly across specificity groups and
    model arms. Higher and lower strata can contain different informative units;
    the exact shared-unit result is exported separately as a small sensitivity.
    """
    q=q.copy()
    if 'component' not in q: q['component']=dependency_components(q)
    y=q.label.to_numpy(); scores=np.abs(data['signed']); rng=np.random.default_rng(seed)
    keys=[r[:2] for r in RUNS]
    unit_rows=[]; summary_rows=[]; difference_rows=[]; model_rows=[]; strict_rows=[]
    saved_draws={}
    for cohort,cfg in cohort_masks(data).items():
        support=cfg['support']; columns=cfg['columns']
        units=[]
        eligible=q[support & q.k562_active.to_numpy(dtype=bool)]
        for (dataset,gene,group),sub in eligible.groupby(['dataset','gene_id','specificity'],sort=False):
            if group not in ['higher','lower'] or sub.label.nunique()!=2: continue
            idx=sub.index.to_numpy()
            assert sub.component.nunique()==1, 'One gene must belong to one dependency component'
            units.append(dict(unit_id=f'{dataset}|{gene}',dataset=dataset,gene_id=gene,
                              specificity=group,component=int(sub.component.iloc[0]),indices=idx,
                              n=len(idx),positives=int(sub.label.sum()),negatives=int((1-sub.label).sum())))
        components=np.unique([u['component'] for u in units])
        component_position={c:i for i,c in enumerate(components)}
        weights=np.vstack([rng.multinomial(len(components),np.full(len(components),1/len(components)))
                           for _ in range(n_boot)]) if len(components) else np.empty((n_boot,0),int)
        by_model_group={}
        for j in columns:
            for group in ['higher','lower']:
                selected=[u for u in units if u['specificity']==group]
                aucs=[]
                for u in selected:
                    auc=roc_auc_score(y[u['indices']],scores[u['indices'],j])
                    aucs.append(auc)
                    unit_rows.append(dict(cohort=cohort,model=RUNS[j][0],arm=RUNS[j][1],AUROC=auc,**{
                        k:u[k] for k in ['unit_id','dataset','gene_id','specificity','component','n','positives','negatives']}))
                ids=[component_position[u['component']] for u in selected]
                draws=_macro_draws(aucs,ids,weights)
                point=float(np.mean(aucs)) if aucs else np.nan
                good=draws[np.isfinite(draws)]
                lo,hi=np.percentile(good,[2.5,97.5]) if len(good) else (np.nan,np.nan)
                by_model_group[(j,group)]=(point,draws)
                summary_rows.append(dict(cohort=cohort,specificity=group,model=RUNS[j][0],arm=RUNS[j][1],
                    macro_AUROC=point,ci_low=lo,ci_high=hi,valid_bootstraps=len(good),
                    n_gene_datasets=len(selected),n_genes=len({u['gene_id'] for u in selected}),
                    n_candidates=sum(u['n'] for u in selected),positives=sum(u['positives'] for u in selected),
                    negatives=sum(u['negatives'] for u in selected)))
            high,low=by_model_group[(j,'higher')],by_model_group[(j,'lower')]
            delta=high[0]-low[0]; samples=high[1]-low[1]; good=samples[np.isfinite(samples)]
            lo,hi=np.percentile(good,[2.5,97.5]) if len(good) else (np.nan,np.nan)
            difference_rows.append(dict(cohort=cohort,model=RUNS[j][0],arm=RUNS[j][1],
                higher_minus_lower=delta,ci_low=lo,ci_high=hi,valid_bootstraps=len(good),
                crosses_zero=bool(lo<=0<=hi) if len(good) else None))
        for left,right in RANK_CONTRASTS:
            a_idx,b_idx=keys.index(left),keys.index(right)
            if a_idx not in columns or b_idx not in columns: continue
            points=[]; samples=[]
            for group in ['higher','lower']:
                pa,da=by_model_group[(a_idx,group)];pb,db=by_model_group[(b_idx,group)]
                points.append(pa-pb);samples.append(da-db)
            contrast=points[0]-points[1]; draws_delta=samples[0]-samples[1]
            good=draws_delta[np.isfinite(draws_delta)];lo,hi=np.percentile(good,[2.5,97.5]) if len(good) else (np.nan,np.nan)
            model_rows.append(dict(cohort=cohort,left=' / '.join(left),right=' / '.join(right),
                higher_gap=points[0],lower_gap=points[1],difference_of_differences=contrast,
                ci_low=lo,ci_high=hi,valid_bootstraps=len(good),crosses_zero=bool(lo<=0<=hi) if len(good) else None))
        # Exact shared gene-dataset units: observed sensitivity only; n is too small for claims.
        high_ids={u['unit_id'] for u in units if u['specificity']=='higher'}
        low_ids={u['unit_id'] for u in units if u['specificity']=='lower'}
        shared=high_ids & low_ids
        for j in columns:
            for group in ['higher','lower']:
                vals=[]
                for u in units:
                    if u['specificity']==group and u['unit_id'] in shared:
                        vals.append(roc_auc_score(y[u['indices']],scores[u['indices'],j]))
                strict_rows.append(dict(cohort=cohort,specificity=group,model=RUNS[j][0],arm=RUNS[j][1],
                    macro_AUROC=float(np.mean(vals)) if vals else np.nan,n_shared_gene_datasets=len(shared)))
        saved_draws[cohort]=weights
        progress('Within-gene ranking',cohort,':',len(units),'informative stratum-units;',len(shared),'shared units')
    units=pd.DataFrame(unit_rows);summary=pd.DataFrame(summary_rows);differences=pd.DataFrame(difference_rows)
    model_differences=pd.DataFrame(model_rows);strict=pd.DataFrame(strict_rows)
    units.to_csv(CACHE/'within_gene_unit_AUROC.tsv',sep='\t',index=False)
    summary.to_csv(CACHE/'within_gene_macro_AUROC.tsv',sep='\t',index=False)
    differences.to_csv(CACHE/'within_gene_specificity_differences.tsv',sep='\t',index=False)
    model_differences.to_csv(CACHE/'within_gene_model_interactions.tsv',sep='\t',index=False)
    strict.to_csv(CACHE/'within_gene_shared_unit_sensitivity.tsv',sep='\t',index=False)
    atomic_json(CACHE/'within_gene_manifest.json',dict(version=1,n_boot=n_boot,seed=seed,
        unit='experimental dataset x target gene x specificity; requires both labels',
        estimand='unweighted macro mean of within-unit AUROC',
        resampling='shared multinomial weights over gene/overlapping-enhancer dependency components',
        primary_limitation='specificity groups can contain different informative gene-dataset units',
        shared_unit_sensitivity='observed macro AUROC only; very small and not used for inference',
        code_sha256=digest(Path(__file__)),source_cache_sha256=digest(CACHE/'complete.json')))
    return summary,differences,model_differences,strict


def plot_within_gene(summary,cohort='common',arm='shuffle',diagnostic=True):
    t=summary[(summary.cohort==cohort)&(summary.arm==arm)]
    models=[m for m in MODELS if m in t.model.unique()]
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharex=True,sharey=True)
    for ax,group in zip(axes,['lower','higher']):
        d=t[t.specificity==group].set_index('model').reindex(models)
        ax.barh(range(len(d)),d.macro_AUROC-.3,left=.3,color=[MODEL_COLORS[m] for m in models],height=.62)
        for i,r in enumerate(d.itertuples()):
            label_x=max(r.macro_AUROC,r.ci_high) if diagnostic else r.macro_AUROC
            ax.text(min(label_x+.008,.97),i,f'{r.macro_AUROC:.3f}',va='center',fontsize=9)
            if diagnostic: ax.plot([r.ci_low,r.ci_high],[i,i],color='#333',lw=1.5)
        if diagnostic: ax.axvline(.5,color='#888',ls=':')
        first=d.iloc[0]
        ax.set_title(f'{group.capitalize()} specificity\n{int(first.n_gene_datasets)} informative gene–dataset units')
        ax.set_xlim(.3,1);ax.set_xlabel('Macro within-gene/dataset AUROC')
        ax.grid(axis='x',alpha=.15);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(range(len(models)),models);axes[0].invert_yaxis()
    fig.suptitle(f'Conditional enhancer ranking · {cohort} · {arm}')
    fig.text(.5,.01,'Each gene–dataset contributes equally. '+
             ('95% component-bootstrap intervals; dotted line = chance.' if diagnostic else 'Point estimates; uncertainty omitted.'),ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);folder=CACHE/'figures';folder.mkdir(exist_ok=True)
    name=f'within_gene_{cohort}_{arm}_{"diagnostic" if diagnostic else "presentation"}'
    for ext in ['png','pdf']:fig.savefig(folder/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    return fig
