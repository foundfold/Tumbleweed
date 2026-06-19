"""Assemble the public Tumbleweed-KdBench release: actual aptamer + K_D + citation.

Emits a clean, citable table where each row is a measured aptamer: sequence, chemistry,
protein target, K_D (nM), and a real primary-literature citation resolved from its PubMed
ID via NCBI E-utilities. Rows without a PubMed ID keep the data and leave citation blank
(per Marko: include them anyway). Aggregator databases (AptaDB/AptamerBase) were used only
as a PMID/citation finder; we credit the original paper, not the aggregator.

Citation source: NCBI E-utilities esummary
  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id=...

Inputs (data_refs/):
  aptamer_kd_all_unified_v3.parquet   curated K_D corpus (911 aptamers)
  recovery_seqs.parquet               RecoveryBench winner/random sequences

Outputs (benchmark_release/):
  kdbench/kdbench_aptamers.csv         aptamer, chemistry, target, K_D, pubmed_id, doi, citation
  kdbench/references.csv               pmid, authors, year, title, journal, doi, url
  kdbench/references.bib               same as BibTeX
  recoverybench/recoverybench_sequences.parquet
  tables/{recoverybench_per_target,kdbench_stability}.csv  (copied)
"""
from __future__ import annotations
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / 'data_refs'
REL = ROOT / 'benchmark_release'
EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'


def clean_pmids(s: pd.Series) -> pd.Series:
    p = s.astype('string').str.strip()
    bad = p.isna() | (p == '') | (p.str.lower() == 'nan') | (p == '0')
    return p.mask(bad, pd.NA)


def fetch_pubmed(pmids: list[str]) -> dict:
    """Resolve PMIDs to citation metadata via NCBI esummary, in batches of 200."""
    out: dict[str, dict] = {}
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i + 200]
        url = EUTILS + '?' + urllib.parse.urlencode(
            {'db': 'pubmed', 'retmode': 'json', 'id': ','.join(batch)})
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                res = json.load(r).get('result', {})
        except Exception as e:
            print(f'  [warn] esummary batch {i} failed: {e}')
            continue
        for pmid in res.get('uids', []):
            rec = res.get(pmid, {})
            authors = rec.get('authors', [])
            first = authors[0]['name'] if authors else ''
            year = (rec.get('pubdate', '') or '')[:4]
            doi = ''
            for aid in rec.get('articleids', []):
                if aid.get('idtype') == 'doi':
                    doi = aid.get('value', '')
            out[pmid] = {
                'pmid': pmid,
                'first_author': first,
                'authors': '; '.join(a['name'] for a in authors),
                'year': year,
                'title': rec.get('title', '').rstrip('.'),
                'journal': rec.get('source', ''),
                'doi': doi,
                'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
            }
        time.sleep(0.4)
    return out


def citation_str(m: dict) -> str:
    bits = []
    if m['first_author']:
        bits.append(f"{m['first_author']} et al.")
    if m['title']:
        bits.append(m['title'] + '.')
    if m['journal']:
        bits.append(m['journal'] + (f" ({m['year']})." if m['year'] else '.'))
    elif m['year']:
        bits.append(f"({m['year']}).")
    bits.append(f"PMID:{m['pmid']}")
    if m['doi']:
        bits.append(f"doi:{m['doi']}")
    return ' '.join(bits)


def bibtex(m: dict) -> str:
    key = (m['first_author'].split(',')[0].split()[0].lower() if m['first_author']
           else 'ref') + (m['year'] or '') + '_' + m['pmid']
    lines = [f'@article{{{key},']
    if m['authors']:
        lines.append(f"  author = {{{m['authors']}}},")
    if m['title']:
        lines.append(f"  title = {{{m['title']}}},")
    if m['journal']:
        lines.append(f"  journal = {{{m['journal']}}},")
    if m['year']:
        lines.append(f"  year = {{{m['year']}}},")
    if m['doi']:
        lines.append(f"  doi = {{{m['doi']}}},")
    lines.append(f"  note = {{PMID:{m['pmid']}}}")
    lines.append('}')
    return '\n'.join(lines)


def main():
    df = pd.read_parquet(DR / 'aptamer_kd_all_unified_v3.parquet')
    df['pubmed_id'] = clean_pmids(df['pubmed_id'])
    unique_pmids = sorted(df['pubmed_id'].dropna().unique().tolist())
    print(f'{len(df)} aptamers; resolving {len(unique_pmids)} unique PubMed IDs ...')
    meta = fetch_pubmed(unique_pmids)
    print(f'resolved {len(meta)}/{len(unique_pmids)} citations')

    df['citation'] = df['pubmed_id'].map(lambda p: citation_str(meta[p]) if p in meta else '')
    df['doi'] = df['pubmed_id'].map(lambda p: meta[p]['doi'] if p in meta else '')

    pub = df[['aptamer_id', 'sequence', 'chemistry_norm', 'target_canonical',
              'target_uniprot_id', 'kd_nm', 'log10_kd_nm',
              'pubmed_id', 'doi', 'citation']].rename(
        columns={'chemistry_norm': 'chemistry', 'target_canonical': 'target'})

    (REL / 'kdbench').mkdir(parents=True, exist_ok=True)
    (REL / 'recoverybench').mkdir(parents=True, exist_ok=True)
    (REL / 'tables').mkdir(parents=True, exist_ok=True)

    pub.to_csv(REL / 'kdbench' / 'kdbench_aptamers.csv', index=False)

    refs = pd.DataFrame([meta[p] for p in unique_pmids if p in meta])
    refs.to_csv(REL / 'kdbench' / 'references.csv', index=False)
    (REL / 'kdbench' / 'references.bib').write_text(
        '\n\n'.join(bibtex(meta[p]) for p in unique_pmids if p in meta) + '\n')

    rec_src = DR / 'recovery_seqs.parquet'
    if rec_src.exists():
        shutil.copy(rec_src, REL / 'recoverybench' / 'recoverybench_sequences.parquet')
    for t in ('recoverybench_per_target.csv', 'kdbench_stability.csv'):
        src = ROOT / 'research' / 'manuscript' / 'tables' / t
        if src.exists():
            shutil.copy(src, REL / 'tables' / t)

    n_cited = int((pub['citation'] != '').sum())
    print(f'wrote {REL/"kdbench"/"kdbench_aptamers.csv"}  '
          f'({len(pub)} aptamers, {n_cited} with citation, {len(pub)-n_cited} without)')
    print('wrote', REL / 'kdbench' / 'references.csv', f'({len(refs)} refs)')
    print('wrote', REL / 'kdbench' / 'references.bib')


if __name__ == '__main__':
    main()
