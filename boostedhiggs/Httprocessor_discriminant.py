from functools import partial
import numpy as np
from coffea import processor, hist
from uproot_methods import TLorentzVectorArray
import awkward
import awkward as ak
from copy import deepcopy
from .common import (
    getBosons,
    matchedBosonFlavor,
    getHTauTauDecayInfo,
    isOverlap,
)
from .corrections import (
    corrected_msoftdrop,
    n2ddt_shift,
    add_pileup_weight,
    add_VJets_NLOkFactor,
    add_jetTriggerWeight,
    add_TriggerWeight,
)
#from .btag import BTagEfficiency, BTagCorrector

# for old pancakes
from coffea.nanoaod.methods import collection_methods, FatJet
collection_methods['CustomAK8Puppi'] = FatJet
collection_methods['CustomAK8PuppiSubjet'] = FatJet
FatJet.subjetmap['CustomAK8Puppi'] = 'CustomAK8PuppiSubjet'

class HtautauProcessorDiscriminant(processor.ProcessorABC):
    def __init__(self, year='2017'):
        self._year = year

        #self._btagSF = BTagCorrector(year, 'medium')
        self._btagWPs = {
            'medium': {
                '2016': 0.6321,
                '2017': 0.4941,
                '2018': 0.4184,
            },
        }

        self._metFilters = {
            '2016': [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
            ],
            '2017': [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
                "BadChargedCandidateFilter",
                "eeBadScFilter",
                "ecalBadCalibFilter",
            ],
            '2018': [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
                "BadChargedCandidateFilter",
                "eeBadScFilter",
                "ecalBadCalibFilterV2",
            ],
        }

        self._hadel_triggers = {
            '2016': [
                #'Ele35_WPTight_Gsf',
'Ele50_CaloIdVT_GsfTrkIdT_PFJet165','Ele115_CaloIdVT_GsfTrkIdT',
#"Ele15_IsoVVVL_PFHT450_PFMET50",
"Ele15_IsoVVVL_PFHT600",
                'PFHT800',
                'PFHT900',
                'AK8PFJet360_TrimMass30',
                'AK8PFHT700_TrimR0p1PT0p03Mass50',
                'PFHT650_WideJetMJJ950DEtaJJ1p5',
                'PFHT650_WideJetMJJ900DEtaJJ1p5',
                #'AK8DiPFJet280_200_TrimMass30_BTagCSV_p20',
                'PFJet450',
            ],
            '2017': [
                #'Ele35_WPTight_Gsf',
'Ele50_CaloIdVT_GsfTrkIdT_PFJet165','Ele115_CaloIdVT_GsfTrkIdT',
#"Ele15_IsoVVVL_PFHT450_PFMET50",
"Ele15_IsoVVVL_PFHT600",
                #'AK8PFJet330_PFAK8BTagCSV_p17',
                'PFHT1050',
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFJet500',
                'AK8PFJet500',
            ],
            '2018': [
                #'Ele35_WPTight_Gsf',
'Ele50_CaloIdVT_GsfTrkIdT_PFJet165','Ele115_CaloIdVT_GsfTrkIdT',
#"Ele15_IsoVVVL_PFHT450_PFMET50",
"Ele15_IsoVVVL_PFHT600",
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFHT1050',
                'PFJet500',
                'AK8PFJet500',
                # 'AK8PFJet330_PFAK8BTagCSV_p17', not present in 2018D?
                #'AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4',
                #'AK4PFJet30',
            ],
        }

        self._hadmu_triggers = {
            '2016': [
                'Mu50','Mu55',
#"Mu15_IsoVVVL_PFHT450_PFMET50",
"Mu15_IsoVVVL_PFHT600",
                'PFHT800',
                'PFHT900',
                'AK8PFJet360_TrimMass30',
                'AK8PFHT700_TrimR0p1PT0p03Mass50',
                'PFHT650_WideJetMJJ950DEtaJJ1p5',
                'PFHT650_WideJetMJJ900DEtaJJ1p5',
                #'AK8DiPFJet280_200_TrimMass30_BTagCSV_p20',
                'PFJet450',
            ],
            '2017': [
                'Mu50',#'Mu55',
#"Mu15_IsoVVVL_PFHT450_PFMET50",
"Mu15_IsoVVVL_PFHT600",
                #'AK8PFJet330_PFAK8BTagCSV_p17',
                'PFHT1050',
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFJet500',
                'AK8PFJet500',
            ],
            '2018': [
                'Mu50',#'Mu55',
#"Mu15_IsoVVVL_PFHT450_PFMET50",
"Mu15_IsoVVVL_PFHT600",
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFHT1050',
                'PFJet500',
                'AK8PFJet500',
                # 'AK8PFJet330_PFAK8BTagCSV_p17', not present in 2018D?
                #'AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4',
                #'AK4PFJet30',
            ],
        }

        self._hadhad_triggers = {
            '2016': [
                'PFHT800',
                'PFHT900',
                'AK8PFJet360_TrimMass30',
                'AK8PFHT700_TrimR0p1PT0p03Mass50',
                'PFHT650_WideJetMJJ950DEtaJJ1p5',
                'PFHT650_WideJetMJJ900DEtaJJ1p5',
                #'AK8DiPFJet280_200_TrimMass30_BTagCSV_p20',
                'PFJet450',
                'DoubleMediumChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg',
                'DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg',
                'DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg',
                'DoubleMediumChargedIsoPFTau40_Trk1_eta2p1_Reg',
            ],
            '2017': [
                #'AK8PFJet330_PFAK8BTagCSV_p17',
                'PFHT1050',
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFJet500',
                'AK8PFJet500',
                'DoubleMediumChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg',
                'DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg',
                'DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg',
                'DoubleMediumChargedIsoPFTau40_Trk1_eta2p1_Reg',
                'MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1',
                'MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_1pr',
            ],
            '2018': [
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFHT1050',
                'PFJet500',
                'AK8PFJet500',
                # 'AK8PFJet330_PFAK8BTagCSV_p17', not present in 2018D?
                #'AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4',
                #'AK4PFJet30',
                'DoubleMediumChargedIsoPFTauHPS35_Trk1_TightID_eta2p1_Reg',
                'DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg',
                'DoubleMediumChargedIsoPFTauHPS40_Trk1_TightID_eta2p1_Reg',
                'DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1_Reg',
                'MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1',
                'MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_1pr',
                'MediumChargedIsoPFTau200HighPtRelaxedIso_Trk50_eta2p1',
                'MediumChargedIsoPFTau220HighPtRelaxedIso_Trk50_eta2p1',
            ],
        }

        jet_pt_bin = hist.Bin('jet_pt', r'Jet $p_{T}$ [GeV]', 20, 200, 1200)
        jet_eta_bin = hist.Bin('jet_eta', r'Jet $\eta$', 20, -3., 3.)
        jet_hadhad_discriminant_bin = hist.Bin('jet_hadhad_discriminant', r'Jet hadhad discriminant', [0.0, 0.0048295204, 0.015817307, 0.19810599, 0.42477998, 0.9169871, 0.9683698, 0.9944695, 0.99777395, 0.9987676, 1.0])
        jet_hadel_discriminant_bin = hist.Bin('jet_hadel_discriminant', r'Jet hadel discriminant', [0.0, 0.023341782, 0.10331184, 0.62654614, 0.8607262, 0.97956115, 0.9916637, 0.99741036, 0.9997223, 0.9997223, 1.0])
        jet_hadmu_discriminant_bin = hist.Bin('jet_hadmu_discriminant', r'Jet hadmu discriminant', [0.0, 0.0366616, 0.11699345, 0.64388525, 0.87208843, 0.9727404, 0.9840254, 0.99656576, 0.9975733, 0.9975733, 1.0])
        jet_msd_bin = hist.Bin('jet_msd', r'Jet $m_{sd}$ [GeV]', 34, 40, 210.)
        oppbjet_pt_bin = hist.Bin('oppbjet_pt', r'Max opp. deepCSV-bjet $p_{T}$ [GeV]', 20, 0., 500)
        oppbtag_bin = hist.Bin('oppbtag', r'Max opp. deepCSV-b ', 20, 0., 1)
        lep_pt_bin = hist.Bin('lep_pt', r'Lepton $p_{T}$ [GeV]', 40, 0, 800)
        lep_eta_bin = hist.Bin('lep_eta', r'Lepton $\eta$', 20, -3., 3.)
        jet_lsf3_bin = hist.Bin('lsf3', r'Jet LSF$_3$', 20, 0., 1.)
        lep_jet_dr_bin = hist.Bin('lep_jet_dr', r'$\Delta R(jet,lepton)$', 40, 0., 4.)
        #lep_miso_bin = hist.Bin('miso', r'Lepton miniIso', 20, 0., 0.1)
        lep_miso_bin = hist.Bin('miso', r'Lepton miniIso', [0.,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
        jet_jetlep_m_bin = hist.Bin('jetlep_m', r'Jet+lepton $m$ [GeV]', 20, 0, 600.)
        jet_jetmet_m_bin = hist.Bin('jetmet_m', r'Jet+MET $m$ [GeV]', 20, 0, 600.)
        jet_jetlepmet_m_bin = hist.Bin('jetlepmet_m', r'Jet+lepton+MET $m$ [GeV]', 20, 0, 600.)
        met_pt_bin = hist.Bin('met_pt', r'MET [GeV]', 20, 0, 800)
        h_pt_bin = hist.Bin('h_pt', r'h $p_{T}$ [GeV]', 20, 200, 1200)
        genhtt_bin = hist.Bin('genhtt',r'hh,eh,mh,em,ee,mm (- for dr > 0.8)',13,-6.5,6.5)
        gentau1had_bin = hist.Bin('gentau1had',r'1pr,1pr+pi0,3pr',4,-0.5,3.5)
        gentau2had_bin = hist.Bin('gentau2had',r'1pr,1pr+pi0,3pr',4,-0.5,3.5)

        self._accumulator = processor.dict_accumulator({
            # dataset -> sumw
            'sumw': processor.defaultdict_accumulator(float),
            # dataset -> cut -> count
            'cutflow_hadhad': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            'cutflow_hadhad_cr_mu': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            'cutflow_hadel': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            'cutflow_hadmu': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            'cutflow_hadel_cr_b': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            'cutflow_hadmu_cr_b': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            'cutflow_hadel_cr_qcd': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            'cutflow_hadmu_cr_qcd': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            #'btagWeight': hist.Hist('Events', hist.Cat('dataset', 'Dataset'), hist.Bin('val', 'BTag correction', 50, 0, 2)), #FIXME
            'jet_kin': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                jet_pt_bin, jet_eta_bin, jet_msd_bin
            ),
            'jet_hadhad': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                jet_pt_bin, jet_hadhad_discriminant_bin, jet_msd_bin
            ),
            'jet_hadel': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                jet_pt_bin, jet_hadel_discriminant_bin, jet_msd_bin
            ),
            'jet_hadmu': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                jet_pt_bin, jet_hadmu_discriminant_bin, jet_msd_bin
            ),
            'b_kin': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                jet_pt_bin, oppbjet_pt_bin, oppbtag_bin,
            ),
            'lep_kin': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                lep_pt_bin, lep_jet_dr_bin, lep_miso_bin,
            ),
            'mass_kin': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                jet_pt_bin, jet_msd_bin, genhtt_bin, #jet_jetmet_m_bin, jet_jetlepmet_m_bin,
            ),
            'evt_kin': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                met_pt_bin, lep_pt_bin, jet_pt_bin, #h_pt_bin,
            ),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata['dataset']
        isRealData = 'genWeight' not in events.columns
        selection = processor.PackedSelection()
        weights = processor.Weights(len(events))
        output = self.accumulator.identity()
        if not isRealData:
            output['sumw'][dataset] += events.genWeight.sum()

        trigger_hadhad = np.zeros(events.size, dtype='bool')
        for t in self._hadhad_triggers[self._year]:
            trigger_hadhad = trigger_hadhad | events.HLT[t]

        trigger_hadmu = np.zeros(events.size, dtype='bool')
        for t in self._hadmu_triggers[self._year]:
            trigger_hadmu = trigger_hadmu | events.HLT[t]

        trigger_hadel = np.zeros(events.size, dtype='bool')
        for t in self._hadel_triggers[self._year]:
            trigger_hadel = trigger_hadel | events.HLT[t]
        #print(np.histogram(trigger))

        if (isRealData): overlap_removal = isOverlap(events,dataset,self._hadhad_triggers[self._year]+self._hadmu_triggers[self._year]+self._hadel_triggers[self._year])
        else: overlap_removal = np.ones(events.size, dtype='bool')

        met_filters = np.ones(events.size, dtype='bool')
        for t in self._metFilters[self._year]:
            met_filters = met_filters & events.Flag[t]

        selection.add('hadhad_trigger', trigger_hadhad & overlap_removal & met_filters)
        selection.add('hadmu_trigger', trigger_hadmu & overlap_removal & met_filters)
        selection.add('hadel_trigger', trigger_hadel & overlap_removal & met_filters)

        try:
            fatjets = events.FatJet
        except AttributeError:
            # early pancakes
            fatjets = events.CustomAK8Puppi
        fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
        fatjets['rho'] = 2 * np.log(fatjets.msdcorr / fatjets.pt)

        candidatejets = fatjets[
            # https://github.com/DAZSLE/BaconAnalyzer/blob/master/Analyzer/src/VJetLoader.cc#L269
            (fatjets.pt > 300)
            & (abs(fatjets.eta) < 2.5)
            & (fatjets.isTight)
        ]#[:, :2]

        met_p4 = TLorentzVectorArray.from_ptetaphim(awkward.JaggedArray.fromiter([[v] for v in events.MET.pt]), awkward.JaggedArray.fromiter([[v] for v in np.zeros(events.size)]), awkward.JaggedArray.fromiter([[v] for v in events.MET.phi]), awkward.JaggedArray.fromiter([[v] for v in np.zeros(events.size)]))
        ak8_met_pair = candidatejets.cross(met_p4)
        ak8_met_dphi = abs(ak8_met_pair.i0.delta_phi(ak8_met_pair.i1))
        #aligned_jet = ak8_met_dphi == ak8_met_dphi.min()
        #best_jet_idx = (ak8_met_pair.i0 + aligned_jet * ak8_met_pair.i1).pt.argmax()
        best_jet_idx = ak8_met_dphi.argmin()
        #best_jet_idx = candidatejets.pt.argmax()
        candidatejet = candidatejets[best_jet_idx]
        candidatejet_rho = 2 * np.log(candidatejet.msdcorr / candidatejet.pt)
        selection.add('jetacceptance', (
            (candidatejet.pt > 300)
            & (candidatejet.msdcorr > 40.)
            & (abs(candidatejet.eta) < 2.4)
            & (candidatejet_rho > -6.)
            & (candidatejet_rho < -2.1)
        ).any())
        selection.add('jetid', (candidatejet.isTight).any())
        #print(np.histogram(candidatejet.pt.fillna(0).flatten()))





        # Discriminate jet selection

        nn_disc_hadhad = awkward.JaggedArray.fromiter([[v] for v in events.IN.hadhad_v4p1])[
            candidatejet.pt.pad(1, clip=True).fillna(0.) > 300.]

        nn_disc_hadmu = awkward.JaggedArray.fromiter([[v] for v in events.GRU.hadmu_v6p1])[
            candidatejet.pt.pad(1, clip=True).fillna(0.) > 300.]

        nn_disc_hadel = awkward.JaggedArray.fromiter([[v] for v in events.GRU.hadel_v6p1])[
            candidatejet.pt.pad(1, clip=True).fillna(0.) > 300.]

        '''
        selectionArray = awkward.firsts(candidatejets)
        
        IN = events.IN
        IN = ak.singletons(IN)
        IN = IN[
            (selectionArray != None)
        ]
        '''


        jets = events.Jet[
            (events.Jet.pt > 30.)
            & (abs(events.Jet.eta) < 2.5)
            & (events.Jet.isTight)
        ]
        # only consider first 4 jets to be consistent with old framework
        jets = jets[:, :4]
        ak4_ak8_pair = jets.cross(candidatejet, nested=True)
        dphi = abs(ak4_ak8_pair.i0.delta_phi(ak4_ak8_pair.i1))
        ak4_opposite = jets[(dphi > np.pi / 2).all()]
        #selection.add('antiak4btagMediumOppHem', ak4_opposite.btagDeepB.max() < BTagEfficiency.btagWPs[self._year]['medium'])
        selection.add('antiak4btagMediumOppHem', ak4_opposite.btagDeepB.max() < self._btagWPs['medium'][self._year])
        ak4_away = jets[(dphi > 0.8).all()]
        #selection.add('ak4btagMedium08', ak4_away.btagDeepB.max() > BTagEfficiency.btagWPs[self._year]['medium'])
        selection.add('ak4btagMedium08', ak4_away.btagDeepB.max() > self._btagWPs['medium'][self._year])

        selection.add('met', events.MET.pt > 50.)
        selection.add('methard', events.MET.pt > 150.)

        el_loose_cuts = [(np.bitwise_and(np.right_shift(events.Electron.vidNestedWPBitmap,events.Electron.vidNestedWPBitmap.ones_like()*(3*k)),events.Electron.vidNestedWPBitmap.ones_like()*7) >= events.Electron.LOOSE) for k in range(10) if k != 7]
        el_tight_cuts = [(np.bitwise_and(np.right_shift(events.Electron.vidNestedWPBitmap,events.Electron.vidNestedWPBitmap.ones_like()*(3*k)),events.Electron.vidNestedWPBitmap.ones_like()*7) >= events.Electron.TIGHT) for k in range(10) if k != 7]
        #el_veto_cuts = [(np.bitwise_and(np.right_shift(events.Electron.vidNestedWPBitmap,events.Electron.vidNestedWPBitmap.ones_like()*(3*k)),events.Electron.vidNestedWPBitmap.ones_like()*7) >= events.Electron.VETO) for k in range(10) if k != 7]
        #                  (MinPtCut,GsfEleSCEtaMultiRangeCut,GsfEleDEtaInSeedCut,GsfEleDPhiInCut,GsfEleFull5x5SigmaIEtaIEtaCut,GsfEleHadronicOverEMEnergyScaledCut,GsfEleEInverseMinusPInverseCut,GsfEleRelPFIsoScaledCut,GsfEleConversionVetoCut,GsfEleMissingHitsCut)
        #                   0       ,1                       ,2                  ,3              ,4                            ,5                                  ,6                             ,7                      ,8                      ,9

        elmask_loose = el_loose_cuts[0].ones_like().astype(bool)
        for m in el_loose_cuts: elmask_loose = elmask_loose & m
        elmask_tight = el_tight_cuts[0].ones_like().astype(bool)
        for m in el_tight_cuts: elmask_tight = elmask_tight & m
        #elmask_veto = el_veto_cuts[0].ones_like().astype(bool)
        #for m in el_veto_cuts: elmask_veto = elmask_veto & m

        goodmuon = (
            (events.Muon.pt > 25)
            & (np.abs(events.Muon.eta) < 2.4)
            #& (events.Muon.sip3d < 4)
            #& (np.abs(events.Muon.dz) < 0.1)
            #& (np.abs(events.Muon.dxy) < 0.05)
            & (events.Muon.mediumId).astype(bool)
            #& (events.Muon.highPtId).astype(bool)
        )
        ngoodmuons = goodmuon.sum()
        leadingmuon = events.Muon[goodmuon].pad(1, clip=True)

        goodelec = (
            (events.Electron.pt > 25)
            & (abs(events.Electron.eta) < 2.5)
            #& (events.Electron.cutBased >= events.Electron.TIGHT)
            #& (events.Electron.cutBased_HEEP).astype(bool)
            & elmask_tight
        )
        ngoodelecs = goodelec.sum()
        leadingelec = events.Electron[goodelec].pad(1, clip=True)

        nmuons = (
            (events.Muon.pt > 15)
            & (abs(events.Muon.eta) < 2.4)
            #& (events.Muon.pfRelIso04_all < 0.25)
            #& (np.abs(events.Muon.dz) < 0.1)
            #& (np.abs(events.Muon.dxy) < 0.05)
            & (events.Muon.looseId).astype(bool)
            #& (events.Muon.highPtId).astype(bool)
        ).sum()

        nelectrons = (
            (events.Electron.pt > 15)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= events.Electron.VETO)
            #& (events.Electron.cutBased_HEEP).astype(bool)
            #& elmask_loose
        ).sum()

        #ntaus = (
        #    (events.Tau.pt > 20)
        #    & (events.Tau.idDecayMode).astype(bool)
        #    # bacon iso looser than Nano selection
        #).sum()
        ntaus = np.zeros(events.size, dtype='bool')

        lepsel = ((nmuons <= 1) & (nelectrons == 0) & (ntaus == 0) & (ngoodelecs == 0) & (ngoodmuons == 1)) | ((nmuons == 0) & (nelectrons <= 1) & (ntaus == 0) & (ngoodmuons == 0) & (ngoodelecs == 1))
        mu_p4 = TLorentzVectorArray.from_ptetaphim(leadingmuon.pt.fillna(0)*lepsel,leadingmuon.eta.fillna(0)*lepsel,leadingmuon.phi.fillna(0)*lepsel,leadingmuon.mass.fillna(0)*lepsel)
#[(goodmuon & ((nmuons == 1) & (nelectrons == 0) & (ntaus == 0) & (ngoodmuons == 1)))]
        muon_ak8_pair = mu_p4.cross(candidatejet, nested=True)
        el_p4 = TLorentzVectorArray.from_ptetaphim(leadingelec.pt.fillna(0)*lepsel,leadingelec.eta.fillna(0)*lepsel,leadingelec.phi.fillna(0)*lepsel,leadingelec.mass.fillna(0)*lepsel)
#[(goodelec & ((nmuons == 0) & (nelectrons == 1) & (ntaus == 0) & (ngoodelecs == 1)))]
        elec_ak8_pair = el_p4.cross(candidatejet, nested=True)
        #leadinglep = awkward.concatenate([mu_p4, el_p4], axis=1).pad(1, clip=True)
        leadinglep = mu_p4 + el_p4

        mu_miso = leadingmuon.miniPFRelIso_all.fillna(0)*lepsel
        el_miso = leadingelec.miniPFRelIso_all.fillna(0)*lepsel
        leadinglep_miso = mu_miso + el_miso
        leadinglep_miso = leadinglep_miso.pad(1, clip=True)

        selection.add('noleptons', (nmuons == 0) & (nelectrons == 0) & (ntaus == 0) & (ngoodmuons == 0) & (ngoodelecs == 0))
        selection.add('onemuon', (nmuons <= 1) & (nelectrons == 0) & (ntaus == 0) & (ngoodelecs == 0) & (ngoodmuons == 1))
        selection.add('oneelec', (nmuons == 0) & (nelectrons <= 1) & (ntaus == 0) & (ngoodmuons == 0) & (ngoodelecs == 1))
        selection.add('muonkin', (
            (leadingmuon.pt > 25.)
            & (abs(leadingmuon.eta) < 2.1)
        ).all())
        selection.add('muonkinhard', (
            (leadingmuon.pt > 60.)
            & (abs(leadingmuon.eta) < 2.1)
        ).all())
        selection.add('muonDphiAK8', (
            abs(muon_ak8_pair.i0.delta_phi(muon_ak8_pair.i1)) > 2*np.pi/3
        ).all().all())
        selection.add('eleckin', (
            (leadingelec.pt > 25.)
            & (abs(leadingelec.eta) < 2.4)
        ).all())
        selection.add('eleckinhard', (
            (leadingelec.pt > 60.)
            & (abs(leadingelec.eta) < 2.4)
        ).all())
        selection.add('elecDphiAK8', (
            abs(elec_ak8_pair.i0.delta_phi(elec_ak8_pair.i1)) > 2*np.pi/3
        ).all().all())

        lep_ak8_pair = leadinglep.cross(candidatejet)#, nested=True)
        selection.add('lepDrAK8', (
            (lep_ak8_pair.i0.delta_r(lep_ak8_pair.i1) < 0.8).all()
            #(lep_ak8_pair.i0.delta_r(lep_ak8_pair.i1) < 99.0).all()
        ))

        #selection.add('jetlsf', (
        #    (candidatejet.lsf3 > 0.7).any()
        #))

        selection.add('miniIso', (
            (leadinglep_miso < 0.1).any()
        ))
        selection.add('miniIsoInv', (
            (leadinglep_miso >= 0.1).any()
        ))

        jet_lep_p4 = lep_ak8_pair.i0 + lep_ak8_pair.i1
        met_jl_pair = met_p4.cross(jet_lep_p4)#, nested=True)
        jet_lep_met_p4 = met_jl_pair.i0 + met_jl_pair.i1
        jet_met_p4 = ak8_met_pair.i0[best_jet_idx] + ak8_met_pair.i1[best_jet_idx]

        if isRealData:
            genflavor = candidatejet.pt.zeros_like()
            w_hadhad = deepcopy(weights)
            w_hadel = deepcopy(weights)
            w_hadmu = deepcopy(weights)
            genHTauTauDecay = candidatejet.pt.zeros_like()
            genHadTau1Decay = candidatejet.pt.zeros_like()
            genHadTau2Decay = candidatejet.pt.zeros_like()
            genHadTau2Decay = candidatejet.pt.zeros_like()
            gentautaudecay = candidatejet.pt.zeros_like()
        else:
            weights.add('genweight', events.genWeight)
            add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
            bosons = getBosons(events)
            genBosonPt = bosons.pt.pad(1, clip=True).fillna(0)
            add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)
            genflavor = matchedBosonFlavor(candidatejet, bosons)
            genHTauTauDecay, genHadTau1Decay, genHadTau2Decay = getHTauTauDecayInfo(events)
            gentautaudecay = awkward.JaggedArray.fromiter([[v] for v in genHTauTauDecay])
            w_hadhad = deepcopy(weights)
            w_hadel = deepcopy(weights)
            w_hadmu = deepcopy(weights)
            #add_TriggerWeight(w_hadhad, candidatejet.msdcorr, candidatejet.pt, leadinglep.pt, self._year, "hadhad")
            #add_TriggerWeight(w_hadel, candidatejet.msdcorr, candidatejet.pt, leadinglep.pt, self._year, "hadel")
            #add_TriggerWeight(w_hadmu, candidatejet.msdcorr, candidatejet.pt, leadinglep.pt, self._year, "hadmu")
            #output['btagWeight'].fill(dataset=dataset, val=self._btagSF.addBtagWeight(weights, ak4_away)) #FIXME

        regions = {
            'hadhad_signal': ['jetacceptance', 'hadhad_trigger', 'jetid', 'antiak4btagMediumOppHem', 'met', 'noleptons'],
            'hadhad_cr_mu': ['jetacceptance', 'hadmu_trigger', 'jetid', 'met', 'ak4btagMedium08', 'onemuon', 'muonkinhard', 'muonDphiAK8'],#,'jetlsf'],
            'hadmu_signal': ['jetacceptance', 'hadmu_trigger', 'jetid', 'antiak4btagMediumOppHem', 'met', 'onemuon', 'muonkin', 'lepDrAK8', 'miniIso'],#, 'jetlsf'],
            'hadel_signal': ['jetacceptance', 'hadel_trigger', 'jetid', 'antiak4btagMediumOppHem', 'met', 'oneelec', 'eleckin', 'lepDrAK8', 'miniIso'],#, 'jetlsf'],
            'hadmu_cr_qcd': ['jetacceptance', 'hadmu_trigger', 'jetid', 'antiak4btagMediumOppHem', 'met', 'onemuon', 'muonkin', 'lepDrAK8', 'miniIsoInv'],#,'jetlsf'],
            'hadel_cr_qcd': ['jetacceptance', 'hadel_trigger', 'jetid', 'antiak4btagMediumOppHem', 'met', 'oneelec', 'eleckin', 'lepDrAK8', 'miniIsoInv'],#,'jetlsf'],
            'hadmu_cr_b': ['jetacceptance', 'hadmu_trigger', 'jetid', 'met', 'onemuon', 'muonkin', 'lepDrAK8', 'miniIso'],#,'jetlsf'],
            'hadel_cr_b': ['jetacceptance', 'hadel_trigger', 'jetid', 'met', 'oneelec', 'eleckin', 'lepDrAK8', 'miniIso'],#,'jetlsf'],
            #'noselection': [],
        }
        w_dict = {
            'hadhad_signal': w_hadhad,
            'hadhad_cr_mu': w_hadmu,
            'hadmu_signal': w_hadmu,
            'hadel_signal': w_hadel,
            'hadmu_cr_qcd': w_hadmu,
            'hadel_cr_qcd': w_hadel,
            'hadmu_cr_b': w_hadmu,
            'hadel_cr_b': w_hadel,
        }

        allcuts_hadel = set()
        allcuts_hadmu = set()
        allcuts_hadel_cr_b = set()
        allcuts_hadmu_cr_b = set()
        allcuts_hadel_cr_qcd = set()
        allcuts_hadmu_cr_qcd = set()
        allcuts_hadhad = set()
        allcuts_hadhad_cr_mu = set()
        output['cutflow_hadel'][dataset]['none'] += float(w_dict['hadel_signal'].weight().sum())
        output['cutflow_hadmu'][dataset]['none'] += float(w_dict['hadmu_signal'].weight().sum())
        output['cutflow_hadel_cr_b'][dataset]['none'] += float(w_dict['hadel_cr_b'].weight().sum())
        output['cutflow_hadmu_cr_b'][dataset]['none'] += float(w_dict['hadmu_cr_b'].weight().sum())
        output['cutflow_hadel_cr_qcd'][dataset]['none'] += float(w_dict['hadel_cr_qcd'].weight().sum())
        output['cutflow_hadmu_cr_qcd'][dataset]['none'] += float(w_dict['hadmu_cr_qcd'].weight().sum())
        output['cutflow_hadhad'][dataset]['none'] += float(w_dict['hadhad_signal'].weight().sum())
        output['cutflow_hadhad_cr_mu'][dataset]['none'] += float(w_dict['hadhad_cr_mu'].weight().sum())
        for cut in regions['hadel_signal']:
            allcuts_hadel.add(cut)
            output['cutflow_hadel'][dataset][cut] += float(w_dict['hadel_signal'].weight()[selection.all(*allcuts_hadel)].sum())
        for cut in regions['hadmu_signal']:
            allcuts_hadmu.add(cut)
            output['cutflow_hadmu'][dataset][cut] += float(w_dict['hadmu_signal'].weight()[selection.all(*allcuts_hadmu)].sum())
        for cut in regions['hadel_cr_b']:
            allcuts_hadel_cr_b.add(cut)
            output['cutflow_hadel_cr_b'][dataset][cut] += float(w_dict['hadel_cr_b'].weight()[selection.all(*allcuts_hadel_cr_b)].sum())
        for cut in regions['hadmu_cr_b']:
            allcuts_hadmu_cr_b.add(cut)
            output['cutflow_hadmu_cr_b'][dataset][cut] += float(w_dict['hadmu_cr_b'].weight()[selection.all(*allcuts_hadmu_cr_b)].sum())
        for cut in regions['hadel_cr_qcd']:
            allcuts_hadel_cr_qcd.add(cut)
            output['cutflow_hadel_cr_qcd'][dataset][cut] += float(w_dict['hadel_cr_qcd'].weight()[selection.all(*allcuts_hadel_cr_qcd)].sum())
        for cut in regions['hadmu_cr_qcd']:
            allcuts_hadmu_cr_qcd.add(cut)
            output['cutflow_hadmu_cr_qcd'][dataset][cut] += float(w_dict['hadmu_cr_qcd'].weight()[selection.all(*allcuts_hadmu_cr_qcd)].sum())
        for cut in regions['hadhad_signal']:
            allcuts_hadhad.add(cut)
            output['cutflow_hadhad'][dataset][cut] += float(w_dict['hadhad_signal'].weight()[selection.all(*allcuts_hadhad)].sum())
        for cut in regions['hadhad_cr_mu']:
            allcuts_hadhad_cr_mu.add(cut)
            output['cutflow_hadhad_cr_mu'][dataset][cut] += float(w_dict['hadhad_cr_mu'].weight()[selection.all(*allcuts_hadhad_cr_mu)].sum())

        systematics = [
            None,
            #'jet_triggerUp',
            #'jet_triggerDown',
            #'btagWeightUp',
            #'btagWeightDown',
            #'btagEffStatUp',
            #'btagEffStatDown',
        ]

        def fill(region, systematic, wmod=None):
            selections = regions[region]
            cut = selection.all(*selections)
            sname = 'nominal' if systematic is None else systematic
            if wmod is None:
                weight = w_dict[region].weight(modifier=systematic)[cut]
            else:
                weight = w_dict[region].weight()[cut] * wmod[cut]

            def normalize(val):
                return val[cut].pad(1, clip=True).fillna(0).flatten()


            #print(dataset)
            #print(region)
            #print("TRIG")
            #print(np.histogram(trigger_hadel[cut]))
            #print("BOSPT")
            #print(np.histogram(normalize(genBosonPt)))
            #print("JETPT")
            #print(np.histogram(normalize(candidatejet.pt)))
            #print("LEPPT")
            #print(np.histogram(normalize(leadinglep.pt)))
            #print("JLDR")
            #print(np.histogram(normalize(lep_ak8_pair.i0.delta_r(lep_ak8_pair.i1))))
            #print("LSF3")
            #print(np.histogram(normalize(candidatejet.lsf3)))
            #print("WEIGHT")
            #print(np.histogram(weight))
            #print("CUTFLOW")
            #print(output['cutflow_hadhad'][dataset])

            '''
            print(dir(events.IN))
            print(dir(events.GRU))
            print(fatjets.pt)
            counter = 0
            for jets in candidatejets.pt:
                    for jet in jets:
                        if jet < 300:
                            counter += 1
            print(counter)
            print(events.IN.hadhad_v4p1)
            print(nn_disc_hadhad)
            print(candidatejets.pt)
            print(len(candidatejets.pt))
            print(len(nn_disc_hadhad))
            '''

            #Cut offs for FPRs of [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]

            #hadhad
            #[0.0048295204, 0.015817307, 0.19810599, 0.42477998, 0.9169871, 0.9683698, 0.9944695, 0.99777395, 0.9987676]

            #hadmu
            #[0.0366616, 0.11699345, 0.64388525, 0.87208843, 0.9727404, 0.9840254, 0.99656576, 0.9975733, 0.9975733]

            #hadel
            #[0.023341782, 0.10331184, 0.62654614, 0.8607262, 0.97956115, 0.9916637, 0.99741036, 0.9997223, 0.9997223]

            output['jet_kin'].fill(
                dataset=dataset,
                region=region,
                jet_pt=normalize(candidatejet.pt),
                jet_eta=normalize(candidatejet.eta),
                jet_msd=normalize(candidatejet.msdcorr),
                weight=weight,
            )

            output['jet_hadhad'].fill(
                dataset=dataset,
                region=region,
                jet_pt=normalize(candidatejet.pt),
                jet_hadhad_discriminant=normalize(nn_disc_hadhad),
                jet_msd=normalize(candidatejet.msdcorr),
                weight=weight,
            )

            output['jet_hadel'].fill(
                dataset=dataset,
                region=region,
                jet_pt=normalize(candidatejet.pt),
                jet_hadel_discriminant=normalize(nn_disc_hadel),
                jet_msd=normalize(candidatejet.msdcorr),
                weight=weight,
            )

            output['jet_hadmu'].fill(
                dataset=dataset,
                region=region,
                jet_pt=normalize(candidatejet.pt),
                jet_hadmu_discriminant=normalize(nn_disc_hadmu),
                jet_msd=normalize(candidatejet.msdcorr),
                weight=weight,
            )

            bmaxind = ak4_opposite.btagDeepB.argmax()
            output['b_kin'].fill(
                dataset=dataset,
                region=region,
                jet_pt=normalize(candidatejet.pt),
                oppbjet_pt=normalize(ak4_opposite[bmaxind].pt),
                oppbtag=normalize(ak4_opposite[bmaxind].btagDeepB),
                weight=weight,
            )

            output['lep_kin'].fill(
                dataset=dataset,
                region=region,
                lep_pt=normalize(leadinglep.pt),
                #lep_eta=normalize(leadinglep.eta),
                #lsf3=normalize(candidatejet.lsf3),
                lep_jet_dr=normalize(lep_ak8_pair.i0.delta_r(lep_ak8_pair.i1)),
                miso=normalize(leadinglep_miso),
                weight=weight,
            )

            output['mass_kin'].fill(
                dataset=dataset,
                region=region,
                jet_pt=normalize(candidatejet.pt),
                jet_msd=normalize(candidatejet.msdcorr),
                genhtt=normalize(gentautaudecay),
                #jetlep_m=normalize(jet_lep_p4.mass),
                #jetmet_m=normalize(jet_met_p4.mass),
                #jetlepmet_m=normalize(jet_lep_met_p4.mass),
                weight=weight,
            )
            output['evt_kin'].fill(
                dataset=dataset,
                region=region,
                met_pt=normalize(met_p4.pt),
                lep_pt=normalize(leadinglep.pt),
                jet_pt=normalize(candidatejet.pt),
                #h_pt=normalize(bosons[events.GenPart.pdgId==25].pt),
                weight=weight,
            )

        for region in regions:
            for systematic in systematics:
                fill(region, systematic)
        #    if 'GluGluHToTauTau' in dataset:
        #        for i in range(9):
        #            fill(region, 'LHEScale_%d' % i, events.LHEScaleWeight[:, i])
        #        for c in events.LHEWeight.columns[1:]:
        #            fill(region, 'LHEWeight_%s' % c, events.LHEWeight[c])

        return output

    def postprocess(self, accumulator):
        return accumulator
