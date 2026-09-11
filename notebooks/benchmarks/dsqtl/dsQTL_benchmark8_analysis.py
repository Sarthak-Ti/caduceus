"""Headless export of dsQTL_benchmark8 notebook mapping and scoring cells."""
from dsQTL_benchmark8_pipeline import *
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CBP = Path('/data1/lesliec/sarthak/data/chrombpnet/dsqtls.yoruba.lcls.benchmarking.all.tsv')
ARMS = [
 ('cis-Mamba (GM12878 DNase logFC)','ours','nomlm_maskonly.npy'),
 ('Enformer (pretrained)','enformer','enformer_dsqtl.npy'),
 ('Enformer retrained (GM12878)','enformer','enformer_dsqtl_retrained-dsqtl.npy'),
 ('Borzoi (pretrained)','borzoi','borzoi_dsqtl_pretrained_1288.npy'),
 ('Borzoi LoRA (GM12878)','borzoi','borzoi_dsqtl_lorav2.npy'),
 ('Borzoi full FT (GM12878)','borzoi','borzoi_dsqtl_trunk.npy'),
 ('Borzoi from scratch (GM12878)','borzoi','borzoi_dsqtl_scratch.npy'),
 ('AlphaGenome','alphagenome','alphagenome_dsqtl.zarr'),
 ('ChromBPNet (GM12878 DNase logFC)','chrombpnet',('encsr000emt','logfc')),
 ('ChromBPNet (GM12878 DNase IPS)','chrombpnet',('encsr000emt','ips')),
 ('ChromBPNet (GM12878 ATAC logFC)','chrombpnet',('encsr637xsc','logfc')),
 ('ChromBPNet (GM12878 ATAC IPS)','chrombpnet',('encsr637xsc','ips')),
]

def map_variants():
    q = pd.read_csv(DSQ/'filtered_dsqtls.txt',sep='\t')
    q.insert(0,'variant_row',np.arange(len(q)))
    q['label_binary'] = (q.label==1).astype(int)
    assert set(q.label.unique()) <= {-1,0,1} and not q.SNPname2.duplicated().any()
    destination = CACHE/'variants_hg38.tsv.gz'
    fingerprint = dict(variants_sha256=digest(DSQ/'filtered_dsqtls.txt'),mapping='prediction scripts rsID -> hg38 BIM, bp minus 1',
                       bims=[dict(path=str(BIMS/f'1000G.EUR.QC.{c}.bim'),size=(BIMS/f'1000G.EUR.QC.{c}.bim').stat().st_size,
                                  mtime_ns=(BIMS/f'1000G.EUR.QC.{c}.bim').stat().st_mtime_ns) for c in range(1,23)])
    if destination.exists():
        assert json.loads((CACHE/'mapping_manifest.json').read_text())==fingerprint
        return pd.read_csv(destination,sep='\t')
    q['pos0_hg38'] = -1
    q['mapping_status'] = 'missing_rsid'
    q['chrom_hg38'] = q.chrom_hg19
    for ch,rows in q.groupby('chrom_hg19'):
        wanted = set(rows.SNPname2)
        hits=[]
        for block in pd.read_csv(BIMS/f'1000G.EUR.QC.{ch[3:]}.bim',sep='\t',header=None,
                                 names=['chrom','rsid','cm','pos','a1','a2'],
                                 usecols=['rsid','pos','a1','a2'],chunksize=500000):
            hits.append(block[block.rsid.isin(wanted)])
        hit = pd.concat(hits).set_index('rsid')
        assert not hit.index.duplicated().any(), f'Ambiguous BIM rsID in {ch}'
        aligned = hit.reindex(rows.SNPname2)
        found = aligned.pos.notna().to_numpy()
        alleles = ((aligned.a1.to_numpy()==rows.allele1.to_numpy()) & (aligned.a2.to_numpy()==rows.allele2.to_numpy())) | \
                  ((aligned.a2.to_numpy()==rows.allele1.to_numpy()) & (aligned.a1.to_numpy()==rows.allele2.to_numpy()))
        valid = found & alleles
        q.loc[rows.index[found & ~alleles],'mapping_status'] = 'allele_mismatch'
        q.loc[rows.index[valid],'mapping_status'] = 'mapped'
        q.loc[rows.index[valid],'pos0_hg38'] = aligned.pos.to_numpy()[valid].astype(np.int64)-1
    q.to_csv(destination,sep='\t',index=False)
    atomic_json(CACHE/'mapping_manifest.json',fingerprint)
    print('Mapping:',q.mapping_status.value_counts().to_dict(),flush=True)
    return q

def score_variants(q):
    path = CACHE/'model_scores.npz'
    paths = [DSQ/f for _,_,f in ARMS if isinstance(f,str)] + [CBP]
    provenance = dict(version=1,variants_sha256=digest(DSQ/'filtered_dsqtls.txt'),arms=ARMS,
                      files=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths])
    # JSON canonicalization converts tuples consistently on fresh and cached runs.
    provenance = json.loads(json.dumps(provenance))
    if path.exists():
        assert json.loads((CACHE/'scores_manifest.json').read_text())==provenance
        with np.load(path) as d: return d['scores'], d['scorable']
    scores = np.full((len(q),len(ARMS)),np.nan)
    scorable = np.zeros_like(scores,dtype=bool)
    t = pd.read_csv(CBP,sep='\t')
    t['_key'] = t['var.chr'].astype(str)+':'+t['var.pos_hg19'].astype(str)
    # Require duplicate keys to agree for every field used before retaining one.
    needed=['var.allele1','var.allele2','obs.label'] + [f'pred.chrombpnet.{f[0]}.varscore.{f[1]}' for _,_,f in ARMS if isinstance(f,tuple)]
    du=t[t._key.duplicated(keep=False)]
    if len(du): assert du.groupby('_key')[needed].nunique(dropna=False).le(1).all().all()
    r=t.drop_duplicates('_key').set_index('_key').reindex(q.chrom_hg19+':'+q.pos_hg19.astype(str))
    for i,(name,family,f) in enumerate(ARMS):
        if isinstance(f,tuple):
            s=r[f'pred.chrombpnet.{f[0]}.varscore.{f[1]}'].to_numpy(dtype=float)
            ok=np.isfinite(s)
            for cb,qc in [('var.allele1','allele1'),('var.allele2','allele2'),('obs.label','label')]:
                assert np.array_equal(r[cb].to_numpy()[ok],q[qc].to_numpy()[ok])
            scores[:,i]=np.abs(s); scorable[:,i]=ok
        else:
            if f.endswith('.zarr'):
                # AlphaGenome is one variant per chunk; matches the poster's full stored profile.
                m=metadata(DSQ/f)
                assert m['shape'][0]==len(q) and m['chunks']==[1,m['shape'][1],2]
                a=np.empty(m['shape'],dtype=m['dtype'])
                for j in range(len(q)): a[j]=read_chunk(DSQ/f,m,(j,0,0))[0]
            else:
                a=np.load(DSQ/f,mmap_mode='r')
            assert a.ndim==3 and a.shape[0]==len(q) and a.shape[2]==2
            if 'borzoi' in f:
                assert a.shape[1]==6144
                a=a[:,3040:3104,:]
            elif 'enformer' in f:
                assert a.shape[1]==896
                a=a[:,440:456,:]
            a=np.asarray(a,dtype=np.float64)
            assert np.isfinite(a).all() and (a>=0).all(), name
            means=a.mean(axis=1)
            ok=(means[:,0]>0)&(means[:,1]>0)
            s=np.zeros(len(q))
            s[ok]=np.abs(np.log2(means[ok,0]/means[ok,1]))
            scores[:,i]=s; scorable[:,i]=ok & np.isfinite(s)
        print(name, 'scorable',int(scorable[:,i].sum()),flush=True)
    np.savez_compressed(path,scores=scores,scorable=scorable)
    atomic_json(CACHE/'scores_manifest.json',provenance)
    return scores,scorable

def annotate_variants(q):
    """Half-open hg38 intervals; variants outside GM peaks stay explicitly unassigned."""
    out=q.copy()
    out['bin_index']=out.pos0_hg38//BIN
    for kind in ['bins','merged']:
        peaks=pd.read_csv(CACHE/f'gm12878_{kind}.tsv.gz',sep='\t')
        out[f'{kind}_specificity']='outside_GM12878_peaks'
        out[f'{kind}_breadth']=np.nan
        out[f'{kind}_peak_id']=''
        out.loc[out.mapping_status!='mapped',f'{kind}_specificity']='unmapped'
        for ch,rows in out[out.mapping_status=='mapped'].groupby('chrom_hg38'):
            p=peaks[peaks.chrom==ch].sort_values('start')
            if p.empty: continue
            pos=rows.pos0_hg38.to_numpy()
            j=np.searchsorted(p.start.to_numpy(),pos,side='right')-1
            ok=(j>=0)&(pos<p.end.to_numpy()[np.maximum(j,0)])
            rr=rows.index[ok]; pp=p.iloc[j[ok]]
            for suffix in ['specificity','breadth','peak_id']:
                out.loc[rr,f'{kind}_{suffix}']=pp[suffix].to_numpy()
    out.to_csv(CACHE/'variant_specificity.tsv.gz',sep='\t',index=False)
    return out

def metric_values(y,s):
    if len(y)==0 or np.unique(y).size<2: return np.nan,np.nan
    return average_precision_score(y,s),roc_auc_score(y,s)

def stratified_metrics(v,scores,scorable,n_boot=200,seed=42):
    """Cluster bootstrap resamples merged peaks; same draws for all model arms."""
    common=scorable.all(axis=1)
    y=v.label_binary.to_numpy()
    rows=[]; differences=[]
    rng=np.random.default_rng(seed)
    for kind in ['bins','merged']:
        for group in ['all_common','all_GM12878_peaks','high_specificity','low_specificity','outside_GM12878_peaks']:
            if group=='all_common': mask=common.copy()
            elif group=='all_GM12878_peaks': mask=common & v[f'{kind}_specificity'].isin(['high_specificity','low_specificity']).to_numpy()
            else: mask=common & v[f'{kind}_specificity'].eq(group).to_numpy()
            idx=np.flatnonzero(mask)
            # All variants sharing a merged peak remain together, including in bin analysis.
            cl=v.iloc[idx].merged_peak_id.fillna('').to_numpy().astype(str)
            fallback=(v.iloc[idx].chrom_hg38+':'+v.iloc[idx].bin_index.astype(str)).to_numpy()
            cl=np.where(cl=='',fallback,cl)
            unique,inv=np.unique(cl,return_inverse=True)
            samples=[np.flatnonzero(inv==j) for j in range(len(unique))]
            boot=np.full((n_boot,len(ARMS)),np.nan)
            if len(unique)>1 and np.unique(y[idx]).size==2:
                for b in range(n_boot):
                    take=np.concatenate([samples[j] for j in rng.integers(0,len(samples),len(samples))])
                    yy=y[idx[take]]
                    if np.unique(yy).size<2: continue
                    for a in range(len(ARMS)):
                        boot[b,a]=average_precision_score(yy,scores[idx[take],a])
            for a,(name,family,_) in enumerate(ARMS):
                ap,roc=metric_values(y[idx],scores[idx,a])
                good=np.isfinite(boot[:,a])
                ci=np.percentile(boot[good,a],[2.5,97.5]) if good.any() else [np.nan,np.nan]
                prevalence=float(y[idx].mean()) if len(idx) else np.nan
                rows.append(dict(definition=kind,stratum=group,model=name,family=family,n=len(idx),
                                 positives=int(y[idx].sum()),negative=int(len(idx)-y[idx].sum()),n_clusters=len(unique),
                                 prevalence=prevalence,AP=ap,AUROC=roc,AP_minus_prevalence=ap-prevalence,
                                 AP_ci_low=ci[0],AP_ci_high=ci[1],valid_bootstraps=int(good.sum())))
                good=good & np.isfinite(boot[:,0])
                delta=boot[good,0]-boot[good,a]
                ci=np.percentile(delta,[2.5,97.5]) if len(delta) else [np.nan,np.nan]
                base=metric_values(y[idx],scores[idx,0])[0]
                differences.append(dict(definition=kind,stratum=group,reference=ARMS[0][0],model=name,
                                        delta_AP=base-ap,ci_low=ci[0],ci_high=ci[1],valid_bootstraps=len(delta)))
    table=pd.DataFrame(rows); table.to_csv(CACHE/'stratified_metrics.tsv',sep='\t',index=False)
    pd.DataFrame(differences).to_csv(CACHE/'paired_model_differences.tsv',sep='\t',index=False)
    return table

def variant_comparisons(v,scores,scorable):
    common=scorable.all(axis=1)
    rows=[]
    for kind in ['bins','merged']:
        eligible=common & v[f'{kind}_specificity'].isin(['high_specificity','low_specificity']).to_numpy()
        if not eligible.any(): continue
        # Fixed global threshold within GM peaks, shared across the two strata for each model.
        cut=np.quantile(scores[eligible],.9,axis=0)
        top=(scores>=cut) & eligible[:,None]
        for group in ['high_specificity','low_specificity']:
            mask=eligible & v[f'{kind}_specificity'].eq(group).to_numpy()
            pos=mask & v.label_binary.eq(1).to_numpy()
            for a,(name,_,_) in enumerate(ARMS):
                selected=top[:,a]&mask
                rows.append(dict(definition=kind,stratum=group,model=name,threshold=cut[a],
                                 n=int(mask.sum()),n_selected=int(selected.sum()),positives=int(pos.sum()),
                                 positives_selected=int((selected&pos).sum()),
                                 positive_recall=(selected&pos).sum()/pos.sum() if pos.any() else np.nan))
    pd.DataFrame(rows).to_csv(CACHE/'top_decile_recovery.tsv',sep='\t',index=False)
    export=v.copy(); export['common_support']=common
    for a,(name,_,_) in enumerate(ARMS): export[name]=np.where(scorable[:,a],scores[:,a],np.nan)
    export.to_csv(CACHE/'variant_scores_and_specificity.tsv.gz',sep='\t',index=False)

def plot_results(table):
    figures=CACHE/'figures'; figures.mkdir(exist_ok=True)
    colors={'high_specificity':'#2878B5','low_specificity':'#DD8452'}
    for kind in ['bins','merged']:
        fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=True)
        for ax,metric in zip(axes,['AP','AUROC']):
            for offset,group in zip([-.17,.17],colors):
                d=table[(table.definition==kind)&(table.stratum==group)].set_index('model').reindex([a[0] for a in ARMS])
                label=f"{group.replace('_',' ')} (n={int(d.n.iloc[0]):,}, pos={int(d.positives.iloc[0])})"
                ax.barh(np.arange(len(d))+offset,d[metric],height=.32,label=label,color=colors[group])
                if metric=='AP':
                    for j,r in enumerate(d.itertuples()):
                        if np.isfinite(r.AP_ci_low): ax.plot([r.AP_ci_low,r.AP_ci_high],[j+offset]*2,color='#333333',lw=1)
                    ax.axvline(d.prevalence.iloc[0],color=colors[group],ls=':',lw=1)
            if metric=='AUROC': ax.axvline(.5,color='gray',ls=':')
            ax.set_xlabel('Average precision (dotted: group prevalence)' if metric=='AP' else 'AUROC')
            ax.set_xlim(0,1); ax.spines[['top','right']].set_visible(False)
        axes[0].set_yticks(np.arange(len(ARMS)),[a[0] for a in ARMS]); axes[0].invert_yaxis()
        axes[0].legend(fontsize=8,loc='lower right')
        fig.suptitle(f'dsQTL specificity: {kind}; AP intervals resample merged peaks')
        fig.tight_layout();fig.savefig(figures/f'{kind}_model_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    q=map_variants()
    scores,scorable=score_variants(q)
    if (CACHE/'peaks.done.json').exists():
        v=annotate_variants(q)
        table=stratified_metrics(v,scores,scorable)
        variant_comparisons(v,scores,scorable)
        plot_results(table)
        print(table[['definition','stratum','n','positives']].drop_duplicates().to_string(index=False),flush=True)
    else:
        print('Coordinates and scores cached; genome-wide peak build still running.',flush=True)
