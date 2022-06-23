import numpy as np
import pandas as pd
import sys
import os
from argparse import ArgumentParser

def parse_args(args):
    description = 'Convert ccube input to pyclone input.'
    parser = ArgumentParser(description=description)
    parser.add_argument('ccube_input',
                        help='ccube input file path')
    parser.add_argument('titan_input',
                        help='titan SCNA input file (segments) path')
    parser.add_argument('output_type',
                        choices=['pyclone', 'dpclust', 'sciclone'],
                        help='Which clustering software we are creating input for.')
    parser.add_argument('outfile',
                        help='name for pyclone input file')

    return parser.parse_args(args)

def estimate_ccf(vaf, p, cn, cr, cv, bv):
    # translated from ccube's MapVaf2CcfPyClone
    # although similar, we'll use Dentro's calculate_ui
    # divided by multiplicity instead for dpclust input
    un, ur = 0, 0

    if bv == cv:
        uv = 1
    elif bv == 0:
        uv = 0
    else:
        uv = bv / cv

    ccf = ((1 - p) * cn * (un - vaf) + p * cr * (ur - vaf)) / (p * cr * (ur - vaf) - p * cv * (uv - vaf))

    return ccf

def calculate_ui(vaf, p, ct, cn):
    # based on Dentro 2022 equation 4
    ui = vaf * (1 / p) * (p * ct + cn * (1 - p))
    return ui

def estimate_mult(ui):
    # based on Dentro 2022 equations 3
    mult = round(ui) if ui >= 1 else 1
    return mult

def calculate_average_cn(gtype):
    if '|' in gtype:
        # subclonal cnopy-number
        cn_raw = [cn.split(',') for cn in gtype.split('|')]
        cns = [float(gt[0]) + float(gt[1]) * float(gt[2]) for gt in cn_raw]
        return sum(cns)
    else:
        cn_raw = gtype.split(',')
        cns = float(cn_raw[0]) + float(cn_raw[1])
        return cns

# from SVclone/load_data.py
def load_titan(titan_file):
    cnv_df = pd.read_csv(titan_file, sep='\t')

    cnv_df.loc[:, 'Chromosome'] = cnv_df.Chromosome.map(str)
    cnv_df.loc[np.isnan(cnv_df.Cellular_Prevalence), 'Cellular_Prevalence'] = 1

    # TITAN doesn't use the Major/Minor CN fields where
    # we only have a single chromosome (i.e. XY genotype)
    # so we need to set these from the Copy_Number field
    xy = np.isnan(cnv_df.Corrected_MajorCN)
    cnv_df.loc[xy, 'Corrected_MajorCN'] = cnv_df.loc[xy, 'Corrected_Copy_Number'].values
    cnv_df.loc[xy, 'Corrected_MinorCN'] = 0

    gtypes = cnv_df.Corrected_MajorCN.map(str) + ',' + \
             cnv_df.Corrected_MinorCN.map(str) + ',' + \
             cnv_df.Cellular_Prevalence.map(str)

    # join subclonal genotypes
    subclonal = cnv_df.Cellular_Prevalence != 1
    frac2 = 1 - cnv_df.Cellular_Prevalence

    # assume neutral CN, or 1 copy on XY
    xy_sc = np.logical_and(subclonal, xy)
    assumed_gtype = '1.0,1.0,' + frac2.map(str)
    assumed_gtype[xy_sc] = '1.0,0.0,' + frac2[xy_sc].map(str)

    gtypes[subclonal] = gtypes[subclonal] + '|' + \
                        assumed_gtype

    cnv_df.loc[:, 'gtype'] = gtypes
    cnv_df = cnv_df.rename(columns={'Chromosome': 'chr', 'Start': 'startpos', 'End': 'endpos'})
    cnv_df = cnv_df.rename(columns={'Start_Position.bp.': 'startpos', 'End_Position.bp.': 'endpos'}) # alternate names
    select_cols = ['chr', 'startpos', 'endpos', 'gtype']
    return cnv_df[select_cols]

# from SVclone/run_filter.py
def match_snv_copy_numbers(snv_df, cnv_df):
    bp_chroms = np.unique(snv_df['chrom'].values)

    for bchr in bp_chroms:
        gtypes = []
        current_chr = snv_df['chrom'].values==bchr
        var_tmp = snv_df[current_chr]
        cnv_tmp = cnv_df[cnv_df['chr']==bchr]

        if len(cnv_tmp)==0:
            continue

        for pos in var_tmp['pos']:
            cnv_start_list = cnv_tmp.startpos.values
            cnv_end_list   = cnv_tmp.endpos.values
            overlaps = np.logical_and(pos >= cnv_start_list, pos <= cnv_end_list)
            match = cnv_tmp[overlaps]

            if len(match)==0:
                gtypes.append('')
            else:
                gtype = match.loc[match.index[0]].gtype
                gtypes.append(gtype)

        snv_indexes = snv_df[current_chr].index.values
        snv_df.loc[snv_indexes,'gtype'] = gtypes
    return snv_df

def main():
    args = parse_args(sys.argv[1:])
    ccube_infile = args.ccube_input
    titan_infile = args.titan_input
    outfile = args.outfile

    ccube = pd.read_csv(ccube_infile, sep='\t')
    tmp = ccube.mutation_id.str.split('_', 1, expand=True)
    ccube['chrom'] = tmp[0].values
    ccube['pos'] = tmp[1].map(int).values

    titan = load_titan(titan_infile)
    df = match_snv_copy_numbers(ccube, titan)

    if args.output_type == 'pyclone':
        pyclone_in = {
            'mutation_id': df.mutation_id.values,
            'ref_counts': df.ref_counts.map(int).values,
            'var_counts': df.var_counts.map(int).values,
            'normal_cn': 2,
            'minor_cn': df.minor_cn_sub1.map(int).values,
            'major_cn': df.major_cn_sub1.map(int).values,
        }

        pyclone_in = pd.DataFrame.from_dict(pyclone_in)
        pyclone_in = pyclone_in[pyclone_in.major_cn > 0]
        pyclone_in.to_csv(outfile, index=False, sep='\t')

    elif args.output_type == 'dpclust':

        chrs = df.mutation_id.map(lambda x: str(x.split('_')[0]))
        ends = df.mutation_id.map(lambda x: int(x.split('_')[1]))

        wt_counts = df.ref_counts.map(int).values
        mut_counts = df.var_counts.map(int).values
        tcns = df.total_cn_sub1.map(int).values
        vafs = df.vaf.map(float).values

        p = df.purity.map(float).values[0]
        uis = [calculate_ui(v, p, ct, 2) for v, ct in zip(vafs, tcns)]
        mults = [int(estimate_mult(ui)) for ui in uis]
        ccfs = np.array(uis) / np.array(mults)
        #ccfs = [estimate_ccf(v, p, 2, ct, ct, m) for v, ct, m in zip(vafs, tcns, mults)]

        dpclust_in = {
            'chr': chrs,
            'end': ends,
            'WT.count': wt_counts,
            'mut.count': mut_counts,
            'subclonal.CN': tcns,
            'mutation.copy.number': np.array(ccfs) * mults,
            'subclonal.fraction': ccfs,
            'no.chrs.bearing.mut': mults
        }

        dpclust_in = pd.DataFrame.from_dict(dpclust_in)
        dpclust_in = dpclust_in[dpclust_in['subclonal.CN'] > 0]
        dpclust_in.to_csv(outfile, index=False, sep='\t')

    elif args.output_type == 'sciclone':

        chrs = df.mutation_id.map(lambda x: str(x.split('_')[0]))
        pos = df.mutation_id.map(lambda x: int(x.split('_')[1]))

        ref_reads = df.ref_counts.map(int).values
        var_reads = df.var_counts.map(int).values
        vafs = df.vaf.map(float).values

        sciclone_vafs = {
            'chr': chrs,
            'pos': pos,
            'var_reads': var_reads,
            'ref_reads': ref_reads,
            'vaf': vafs * 100
        }

        sciclone_vafs = pd.DataFrame.from_dict(sciclone_vafs)
        sciclone_vafs.to_csv(outfile, index=False, sep='\t')

        cns = titan.gtype.apply(calculate_average_cn)
        sciclone_cns = {
            'chr': titan['chr'].values,
            'start': titan['startpos'].map(int).values,
            'stop': titan['endpos'].map(int).values,
            'segment_mean': cns
        }

        outfile = '%s_cns.txt' % os.path.splitext(args.outfile)[0]
        sciclone_cns = pd.DataFrame.from_dict(sciclone_cns)
        sciclone_cns.to_csv(outfile, index=False, sep='\t')

        cnv_df = pd.read_csv(titan_infile, sep='\t')
        loh = cnv_df[cnv_df.Corrected_Call == 'HETD']
        outfile = '%s_loh.bed' % os.path.splitext(args.outfile)[0]

        if 'Start_Position.bp.' in loh.columns:
            select_columns = ['Chromosome', 'Start_Position.bp.', 'End_Position.bp.']
        elif 'Start' in loh.columns:
            select_columns = ['Chromosome', 'Start', 'End']
        else:
            print('ERROR: Invalid copy-number file', file=sys.stderr)
            sys.exit()

        loh.to_csv(outfile, index=False, sep='\t', columns=select_columns, header=False)


if __name__ == '__main__':
    main()
