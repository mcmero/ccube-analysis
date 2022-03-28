import numpy as np
import pandas as pd
import sys
from argparse import ArgumentParser

def parse_args(args):
    description = 'Convert ccube input to pyclone input.'
    parser = ArgumentParser(description=description)
    parser.add_argument('ccube_input',
                        help='ccube input file path')
    parser.add_argument('titan_input',
                        help='titan SCNA input file (segments) path')
    parser.add_argument('outfile',
                        help='name for pyclone input file')

    return parser.parse_args(args)

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

if __name__ == '__main__':
    main()
