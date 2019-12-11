from coffea import hist

hists_properties = {'jet0_dHqqqq':{'hist_name': 'presel_hadseljet0',
                                   'var_label': r'Jet dak8',
                                   'var_cut_dir': 1,
                                 },
                    'jet0_dHqqqqMD':{'hist_name':'presel_hadseljet0',
                                     'var_label': r'Jet dak8 MD',
                                     'var_cut_dir': 1,
                                     },
                    'jet0_msd':{'hist_name':'presel_hadseljet0',
                                'var_label': r'Jet m$_{SD}$',
                                'var_cut_dir': 1,
                                },
                    'jet0_pt':{'hist_name':'presel_hadseljet0',
                               'var_label': r'Jet $p_T$',
                               'var_cut_dir': 1,
                               },
                    }

process_latex = {'tt': r'$t\bar{t}$',
                 'st': 'Single-t',
                 'zqq': 'Z(qq)',
                 'wqq': 'W(qq)',
                 'qcd': 'Multijet',
                 'wlnu': r'W($\ell\nu$)',
                 'vv': 'Diboson',
                 'h125': 'ggH(125)',
                 'Stat. Unc.': 'Stat. Unc.',
                 }

import re
nosig = re.compile("(?!h125)")
nobkg = re.compile("(?!qcd)(?!tt)(?!st)(?!zqq)(?!wlnu)(?!vv)(?!wqq)")

