from functools import partial
import numpy as np
from coffea import processor, hist
from uproot_methods import TLorentzVectorArray
import awkward
from .common import (
    getBosons,
    matchedBosonFlavor,
)
from .corrections import (
    corrected_msoftdrop,
    n2ddt_shift,
    add_pileup_weight,
    add_VJets_NLOkFactor,
    add_jetTriggerWeight,
)
#from .btag import BTagEfficiency, BTagCorrector

# for old pancakes
from coffea.nanoaod.methods import collection_methods, FatJet
collection_methods['CustomAK8Puppi'] = FatJet
collection_methods['CustomAK8PuppiSubjet'] = FatJet
FatJet.subjetmap['CustomAK8Puppi'] = 'CustomAK8PuppiSubjet'


class HtautauProcessor_Trigger_Fine(processor.ProcessorABC):
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

        self._hadel_triggers = {
            '2016': [
                'Ele35_WPTight_Gsf','Ele50_CaloIdVT_GsfTrkIdT_PFJet165','Ele115_CaloIdVT_GsfTrkIdT',
"Ele15_IsoVVVL_PFHT450_PFMET50",
"Ele15_IsoVVVL_PFHT600",
            ],
            '2017': [
                'Ele35_WPTight_Gsf','Ele50_CaloIdVT_GsfTrkIdT_PFJet165','Ele115_CaloIdVT_GsfTrkIdT',
"Ele15_IsoVVVL_PFHT450_PFMET50",
"Ele15_IsoVVVL_PFHT600",
            ],
            '2018': [
                'Ele35_WPTight_Gsf','Ele50_CaloIdVT_GsfTrkIdT_PFJet165','Ele115_CaloIdVT_GsfTrkIdT',
"Ele15_IsoVVVL_PFHT450_PFMET50",
"Ele15_IsoVVVL_PFHT600",
            ],
        }

        self._hadmu_triggers = {
            '2016': [
                'Mu50','Mu55',
"Mu15_IsoVVVL_PFHT450_PFMET50",
"Mu15_IsoVVVL_PFHT600",
            ],
            '2017': [
                'Mu50','Mu55',
"Mu15_IsoVVVL_PFHT450_PFMET50",
"Mu15_IsoVVVL_PFHT600",
            ],
            '2018': [
                'Mu50','Mu55',
"Mu15_IsoVVVL_PFHT450_PFMET50",
"Mu15_IsoVVVL_PFHT600",
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
                'AK8DiPFJet280_200_TrimMass30_BTagCSV_p20',
                'PFJet450',
            ],
            '2017': [
                'AK8PFJet330_PFAK8BTagCSV_p17',
                'PFHT1050',
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFJet500',
                'AK8PFJet500',
            ],
            '2018': [
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFHT1050',
                'PFJet500',
                'AK8PFJet500',
                # 'AK8PFJet330_PFAK8BTagCSV_p17', not present in 2018D?
                'AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4',
                #'AK4PFJet30',
            ],
        }

        self._all_triggers = {
            '2016': [
                'Ele35_WPTight_Gsf',
                'Ele50_CaloIdVT_GsfTrkIdT_PFJet165',
                'Ele115_CaloIdVT_GsfTrkIdT',
                "Ele15_IsoVVVL_PFHT450_PFMET50",
                "Ele15_IsoVVVL_PFHT600",
                'Mu50',
                'Mu55',
                "Mu15_IsoVVVL_PFHT450_PFMET50",
                "Mu15_IsoVVVL_PFHT600",
                'PFHT800',
                'PFHT900',
                'AK8PFJet360_TrimMass30',
                'AK8PFHT700_TrimR0p1PT0p03Mass50',
                'PFHT650_WideJetMJJ950DEtaJJ1p5',
                'PFHT650_WideJetMJJ900DEtaJJ1p5',
                'AK8DiPFJet280_200_TrimMass30_BTagCSV_p20',
                'PFJet450',
            ],
            '2017': [
                'Mu50',
                'Mu55',
                "Mu15_IsoVVVL_PFHT450_PFMET50",
                "Mu15_IsoVVVL_PFHT600",
                'Ele35_WPTight_Gsf',
                'Ele50_CaloIdVT_GsfTrkIdT_PFJet165',
                'Ele115_CaloIdVT_GsfTrkIdT',
                "Ele15_IsoVVVL_PFHT450_PFMET50",
                "Ele15_IsoVVVL_PFHT600",
                'AK8PFJet330_PFAK8BTagCSV_p17',
                'PFHT1050',
                'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                'AK8PFHT800_TrimMass50',
                'PFJet500',
                'AK8PFJet500',
            ],
            '2018': [
                #'AK8PFJet400_TrimMass30',
                'AK8PFJet420_TrimMass30',
                #'AK8PFHT800_TrimMass50',
                'PFHT1050',
                #'PFJet500',
                'AK8PFJet500',
                #'Ele35_WPTight_Gsf',
                'Ele50_CaloIdVT_GsfTrkIdT_PFJet165',
                #'Ele115_CaloIdVT_GsfTrkIdT',
                "Ele15_IsoVVVL_PFHT450_PFMET50",
                "Ele15_IsoVVVL_PFHT600",
                #'Mu50',
                'Mu55',
                "Mu15_IsoVVVL_PFHT450_PFMET50",
                "Mu15_IsoVVVL_PFHT600",
                "PFMETNoMu110_PFMHTNoMu110_IDTight",
            ],
        }

        self._accumulator = processor.dict_accumulator({
            # dataset -> sumw
            'sumw': processor.defaultdict_accumulator(float),
            # dataset -> cut -> count
            'cutflow_hadhad': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            'cutflow_hadel': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            'cutflow_hadmu': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            #'btagWeight': hist.Hist('Events', hist.Cat('dataset', 'Dataset'), hist.Bin('val', 'BTag correction', 50, 0, 2)), #FIXME
            #'trigeff_h_pt': hist.Hist(
            'trigeff': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('h_pt', r'Higgs $p_{T}$ [GeV]', 10, 300, 800),
                hist.Bin('jet_pt', r'Jet $p_{T}$ [GeV]', 10, 300, 800),
                hist.Bin('lep_pt', r'Lepton $p_{T}$ [GeV]', 10, 0, 100),
                *[hist.Bin('trig_pass_'+t, r'Trigger Pass Bit (%s)'%t, 2, -0.5, 1.5) for t in self._all_triggers[self._year]],
            ),
            #'trigeff_jet_pt': hist.Hist(
            #    'Events',
            #    hist.Cat('dataset', 'Dataset'),
            #    hist.Cat('region', 'Region'),
            #    hist.Bin('jet_pt', r'Jet $p_{T}$ [GeV]', 20, 0, 1000),
            #    *[hist.Bin('trig_pass_'+t, r'Trigger Pass Bit (%s)'%t, 2, -0.5, 1.5) for t in self._all_triggers[self._year]],
            #),
            #'trigeff_lep_pt': hist.Hist(
            #    'Events',
            #    hist.Cat('dataset', 'Dataset'),
            #    hist.Cat('region', 'Region'),
            #    hist.Bin('lep_pt', r'Lepton $p_{T}$ [GeV]', 20, 0, 200),
            #    *[hist.Bin('trig_pass_'+t, r'Trigger Pass Bit (%s)'%t, 2, -0.5, 1.5) for t in self._all_triggers[self._year]],
            #),
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

        trigger_pass = []
        for t in self._all_triggers[self._year]:
            trigger_pass.append(events.HLT[t])

        try:
            fatjets = events.FatJet
        except AttributeError:
            # early pancakes
            fatjets = events.CustomAK8Puppi
        fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
        fatjets['rho'] = 2 * np.log(fatjets.msdcorr / fatjets.pt)
        fatjets['n2ddt'] = fatjets.n2b1 - n2ddt_shift(fatjets, year=self._year)

        candidatejets = fatjets[
            # https://github.com/DAZSLE/BaconAnalyzer/blob/master/Analyzer/src/VJetLoader.cc#L269
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
            & (fatjets.jetId & 2)
        ][:, :2]
        met_p4 = TLorentzVectorArray.from_ptetaphim(awkward.JaggedArray.fromiter([[v] for v in events.MET.phi]), awkward.JaggedArray.fromiter([[v] for v in np.zeros(events.size)]), awkward.JaggedArray.fromiter([[v] for v in events.MET.phi]), awkward.JaggedArray.fromiter([[v] for v in np.zeros(events.size)]))
        ak8_met_pair = candidatejets.cross(met_p4)
        ak8_met_dphi = ak8_met_pair.i0.delta_phi(ak8_met_pair.i1)
        candidatejet = candidatejets[ak8_met_dphi.argmin()]
        selection.add('jetacceptance', (
            (candidatejet.pt > 200)
            & (candidatejet.msdcorr > 50.)
            & (abs(candidatejet.eta) < 2.4)
        ).any())
        selection.add('jetid', (candidatejet.jetId & 2).any())
        selection.add('n2ddt', (candidatejet.n2ddt < 0.).any())
        #print(np.histogram(candidatejet.pt.fillna(0).flatten()))

        jets = events.Jet[
            (events.Jet.pt > 30.)
            & (events.Jet.jetId & 2)
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

        selection.add('met', events.MET.pt > 25.)

        goodmuon = (
            (events.Muon.pt > 25)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.sip3d < 4)
            & (np.abs(events.Muon.dz) < 0.1)
            & (np.abs(events.Muon.dxy) < 0.05)
            & (events.Muon.mediumId).astype(bool)
        )
        ngoodmuons = goodmuon.sum()
        leadingmuon = events.Muon.pad(1, clip=True)

        goodelec = (
            (events.Electron.pt > 35)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= events.Electron.TIGHT)
        )
        ngoodelecs = goodelec.sum()
        leadingelec = events.Electron[goodelec].pad(1, clip=True)

        nmuons = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            #& (events.Muon.pfRelIso04_all < 0.25)
            & (events.Muon.looseId).astype(bool)
        ).sum()

        nelectrons = (
            (events.Electron.pt > 10)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        ).sum()

        #ntaus = (
        #    (events.Tau.pt > 20)
        #    & (events.Tau.idDecayMode).astype(bool)
        #    # bacon iso looser than Nano selection
        #).sum()
        ntaus = np.zeros(events.size, dtype='bool')

        lepsel = ((nmuons == 1) & (nelectrons == 0) & (ntaus == 0) & (ngoodmuons == 1)) | ((nmuons == 0) & (nelectrons == 1) & (ntaus == 0) & (ngoodelecs == 1))
        mu_p4 = TLorentzVectorArray.from_ptetaphim(leadingmuon.pt.fillna(0)*lepsel,leadingmuon.eta.fillna(0)*lepsel,leadingmuon.phi.fillna(0)*lepsel,leadingmuon.mass.fillna(0)*lepsel)
#[(goodmuon & ((nmuons == 1) & (nelectrons == 0) & (ntaus == 0) & (ngoodmuons == 1)))]
        muon_ak8_pair = mu_p4.cross(candidatejet, nested=True)
        el_p4 = TLorentzVectorArray.from_ptetaphim(leadingelec.pt.fillna(0)*lepsel,leadingelec.eta.fillna(0)*lepsel,leadingelec.phi.fillna(0)*lepsel,leadingelec.mass.fillna(0)*lepsel)
#[(goodelec & ((nmuons == 0) & (nelectrons == 1) & (ntaus == 0) & (ngoodelecs == 1)))]
        elec_ak8_pair = el_p4.cross(candidatejet, nested=True)
        #leadinglep = awkward.concatenate([mu_p4, el_p4], axis=1).pad(1, clip=True)
        leadinglep = mu_p4 + el_p4
        lep_ak8_pair = leadinglep.cross(candidatejet)#, nested=True)

        selection.add('noleptons', (nmuons == 0) & (nelectrons == 0) & (ntaus == 0))
        selection.add('onemuon', (nmuons == 1) & (nelectrons == 0) & (ntaus == 0) & (ngoodmuons == 1))
        selection.add('oneelec', (nmuons == 0) & (nelectrons == 1) & (ntaus == 0) & (ngoodelecs == 1))
        selection.add('muonkin', (
            (leadingmuon.pt > 35.)
            & (abs(leadingmuon.eta) < 2.1)
        ).all())
        selection.add('muonDphiAK8', (
            abs(muon_ak8_pair.i0.delta_phi(muon_ak8_pair.i1)) > 2*np.pi/3
        ).all().all())
        selection.add('eleckin', (
            (leadingelec.pt > 35.)
            & (abs(leadingelec.eta) < 2.4)
        ).all())
        selection.add('elecDphiAK8', (
            abs(elec_ak8_pair.i0.delta_phi(elec_ak8_pair.i1)) > 2*np.pi/3
        ).all().all())

        selection.add('lepDrAK8', (
            (lep_ak8_pair.i0.delta_r(lep_ak8_pair.i1) < 0.8).all()
        ))

        if isRealData:
            genflavor = candidatejet.pt.zeros_like()
        else:
            weights.add('genweight', events.genWeight)
            #add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
            bosons = getBosons(events)
            genBosonPt = bosons.pt.pad(1, clip=True).fillna(0)
            #add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)
            genflavor = matchedBosonFlavor(candidatejet, bosons)
            #add_jetTriggerWeight(weights, candidatejet.msdcorr, candidatejet.pt, self._year)
            #output['btagWeight'].fill(dataset=dataset, val=self._btagSF.addBtagWeight(weights, ak4_away)) #FIXME

        regions = {
            'hadhad_signal': ['jetacceptance', 'jetid', 'n2ddt', 'antiak4btagMediumOppHem', 'met', 'noleptons'],
            'hadmu_signal': ['jetacceptance', 'jetid', 'antiak4btagMediumOppHem', 'met', 'onemuon', 'muonkin'],
            'hadel_signal': ['jetacceptance', 'jetid', 'antiak4btagMediumOppHem', 'met', 'oneelec', 'eleckin'],
            #'noselection': [],
        }

        allcuts_hadel = set()
        allcuts_hadmu = set()
        allcuts_hadhad = set()
        output['cutflow_hadel'][dataset]['none'] += float(weights.weight().sum())
        output['cutflow_hadmu'][dataset]['none'] += float(weights.weight().sum())
        output['cutflow_hadhad'][dataset]['none'] += float(weights.weight().sum())
        for cut in regions['hadel_signal']:
            allcuts_hadel.add(cut)
            output['cutflow_hadel'][dataset][cut] += float(weights.weight()[selection.all(*allcuts_hadel)].sum())
        for cut in regions['hadmu_signal']:
            allcuts_hadmu.add(cut)
            output['cutflow_hadmu'][dataset][cut] += float(weights.weight()[selection.all(*allcuts_hadmu)].sum())
        for cut in regions['hadhad_signal']:
            allcuts_hadhad.add(cut)
            output['cutflow_hadhad'][dataset][cut] += float(weights.weight()[selection.all(*allcuts_hadhad)].sum())

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
                weight = weights.weight(modifier=systematic)[cut]
            else:
                weight = weights.weight()[cut] * wmod[cut]

            def normalize(val):
                return val[cut].pad(1, clip=True).fillna(0).flatten()

            tdict = {}
            for it,t in enumerate(self._all_triggers[self._year]):
                tdict['trig_pass_'+t] = trigger_pass[it][cut]

            #output['trigeff_h_pt'].fill(
            output['trigeff'].fill(
                dataset=dataset,
                region=region,
                **tdict,
                h_pt=normalize(genBosonPt),
                jet_pt=normalize(candidatejet.pt),
                lep_pt=normalize(leadinglep.pt),
                #lsf3=normalize(candidatejet.lsf3),
                #weight=weight,
            )
            #output['trigeff_jet_pt'].fill(
            #    dataset=dataset,
            #    region=region,
            #    **tdict,
            #    jet_pt=normalize(candidatejet.pt),
            #    #lsf3=normalize(candidatejet.lsf3),
            #    #weight=weight,
            #)
            #output['trigeff_lep_pt'].fill(
            #    dataset=dataset,
            #    region=region,
            #    **tdict,
            #    lep_pt=normalize(leadinglep.pt),
            #    #lsf3=normalize(candidatejet.lsf3),
            #    #weight=weight,
            #)

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
