import os
import numpy as np
from coffea import processor, util, hist
import json

from boostedhiggs import BTagEfficiency
from coffea.nanoaod import NanoEvents
from boostedhiggs.HttProcessor_discriminant import HtautauProcessorDiscriminant
import argparse


def run_processor(year, selsamples, starti, endi, outname):
    p = HtautauProcessorDiscriminant(year=year)

    files = {}

    with open('../data/fileset2017.json', 'r') as f:
        newfiles = json.load(f)
        files.update(newfiles)

    with open('../data/fileset2017UL.json', 'r') as f:
        newfiles = json.load(f)
        files.update(newfiles)

    with open('../data/fileset2018UL.json', 'r') as f:
        newfiles = json.load(f)
        files.update(newfiles)

    selfiles = {k: files[k][starti:endi] for k in selsamples}

    args = {'nano': True, 'workers': 1, 'savemetrics': True}
    out, metrics = processor.run_uproot_job(selfiles, 'Events', p, processor.futures_executor, args)

    util.save(out, '%s.coffea' % outname)


if __name__ == "__main__":
    # ex. python run_htt.py --year 2018 --starti 0 --endi -1 --selsamples GluGluHToTauTau --outname htt_runtest
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', dest='year', default='2017', help="year", type=str)
    parser.add_argument('--starti', dest='starti', default=0, help="start index", type=int)
    parser.add_argument('--endi', dest='endi', default=-1, help="end index", type=int)
    parser.add_argument('--selsamples', dest='selsamples', default=[], help='selsamples', nargs='+')
    parser.add_argument('--outname', dest='outname', default='htt_test', help='outname')
    args = parser.parse_args()

    run_processor(args.year, args.selsamples, args.starti, args.endi, args.outname)

    possible = [
        'WW_TuneCP5_13TeV-pythia8',
        'WZ_TuneCP5_13TeV-pythia8',
        'ZZ_TuneCP5_13TeV-pythia8',
        'QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8',
        'QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8',
        'QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8',
        'QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8',
        'QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8',
        'QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8',
        'QCD_HT50to100_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8',
        'QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8',
        'DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8',
        'DYJetsToLL_M-50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8',
        'DYJetsToLL_M-50_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8',
        'DYJetsToLL_M-50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8',
        'DYJetsToLL_M-50_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8',
        'DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8',
        'DYJetsToLL_M-50_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8',
        'DYJetsToLL_M-50_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToQQ_HT-800toInf_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToQQ_HT400to600_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8',
        'WJetsToQQ_HT600to800_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8',
        'ZJetsToQQ_HT-800toInf_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8',
        'ZJetsToQQ_HT400to600_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8',
        'ZJetsToQQ_HT600to800_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8',
        'ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8',
        'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8',
        'ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8',
        'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8',
        'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8',
        'ST_tW_top_5f_inclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8',
        'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8',
        'TTToHadronic_TuneCP5_13TeV-powheg-pythia8',
        'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8',
        'GluGluHToTauTau',
    ]
    # TTToHadronic_TuneCP5_13TeV-powheg-pythia8 TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8
    # WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8 WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8 WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8 WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8 WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8
    # QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8 QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8 QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8 QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8 QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8 QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8
# python run_htt_lepid.py --year 2017 --starti 0 --endi 100 --selsamples --outname htt_runtest_lepid_QCD

# print(possible)
#    python run_htt.py --year 2017 --starti 0 --endi 100 --selsamples DYJetsToLL_M-50_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8 --outname htt_runtest
