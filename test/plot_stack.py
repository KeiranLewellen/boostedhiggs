from __future__ import print_function, division
import gzip
import json
import os
import sys

import uproot
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.styles.ROOT)
import numpy as np

from coffea import hist
from coffea.util import load

import pickle
import gzip
import math

import argparse
import processmap
from hists_map import *

plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        #'text.usetex': False,
        })

fill_opts = {
    'edgecolor': (0,0,0,0.3),
    'alpha': 0.8
    }
err_opts = {
    'label':'Stat. Unc.',
    'hatch':'///',
    'facecolor':'none',
    'edgecolor':(0,0,0,.5),
    'linewidth': 0
    }
err_opts_data = {
    'marker': '.',
    'markersize': 10.,
    'color':'k',
    'elinewidth': 1,
    #'emarker': '-'
    }
err_opts_denom = {
    'facecolor':'gray', 
    'alpha':0.6,
    }
line_opts = {
    'color': 'aquamarine',
    'linewidth':2,
    #'marker':'None',
    'linestyle':'dashed',
    'drawstyle':'steps'}

overflow_sum = 'allnan'

def drawStack(h,sel,var_name,var_label,plottitle,lumifb,vars_cut,sig_cut,regionsel,savename,xlimits,blind,solo,sigscale,sigstack,noratio,dosigrat,overflow):
    exceptions = ['process', var_name]
    for var in vars_cut:
        exceptions.append(var)
    for var in sig_cut:
        exceptions.append(var)
    if (regionsel is not ''):
        exceptions.append('region')
    print([ax.name for ax in h.axes()])
    x = h.sum(*[ax for ax in h.axes() if ax.name not in exceptions],overflow=overflow_sum)
    for reg in regionsel:
        print('integrating ',reg)
        x = x.integrate('region',reg)
    for var,val in vars_cut.items():
        if var!=var_name:
            print('integrating ',var,val[0],val[1])
            x = x.integrate(var,slice(val[0],val[1]))
            #x = x.integrate(var,slice(val[0],val[1]),overflow=overflow)
    if var_name in vars_cut.keys():
        x = x[:, vars_cut[var_name][0]:vars_cut[var_name][1]]

    xaxis = var_name
    x.axis(xaxis).label = var_label
    for ih,hkey in enumerate(x.identifiers('process')):
        x.identifiers('process')[ih].label = process_latex[hkey.name]

    x.axis('process').sorting = 'integral'
    if (noratio and not dosigrat): fig,ax = plt.subplots()
    elif (not noratio and dosigrat): fig, (ax,axr,axs) = plt.subplots(3, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1, 1],'hspace': 0.1})
    elif (noratio and dosigrat): fig, (ax,axs) = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1],'hspace': 0.1})
    else: fig, (ax,axr) = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1],'hspace': 0.1})
 
    if (solo):
        hist.plot1d(x,
                overlay='process',ax=ax,
                clear=False,
                stack=False,
                fill_opts=fill_opts,
                error_opts=err_opts,
                overflow=overflow
                )
    else:

        x_nobkg = x[nobkg]

        for var,val in sig_cut.items():
            if var!=var_name:
                print('integrating ',var,val[0],val[1])
                x_nobkg = x_nobkg.integrate(var,slice(val[0],val[1]))
                #x = x.integrate(var,slice(val[0],val[1]),overflow=overflow)
        if var_name in vars_cut.keys():
            x_nobkg = x_nobkg[:, vars_cut[var_name][0]:vars_cut[var_name][1]]
        x = x.sum(*[ax for ax in x.axes() if ax.name in sig_cut],overflow='allnan')

        x_nosig = x[nosig]
        # normalize to lumi
        x_nosig.scale({p: lumifb for p in x_nosig.identifiers('process')}, axis="process")
        all_bkg = 0
        for key,val in x_nosig.values().items():
            all_bkg+=val.sum()
        if (all_bkg>0.): hist.plot1d(x_nosig,
                    overlay='process',ax=ax,
                    clear=False,
                    stack=True,
                    fill_opts=fill_opts,
                    error_opts=err_opts,
                    overflow=overflow
                    )
    
        x_nobkg.scale({p: lumifb*float(sigscale) for p in x_nobkg.identifiers('process')}, axis="process")
    
        all_sig = 0
        for key,val in x_nobkg.values().items():
            all_sig +=val.sum()
        if (all_sig>0.): hist.plot1d(x_nobkg,ax=ax,
                    overlay='process',
                    clear=False,
                    line_opts=line_opts,
                    overflow=overflow)
    
        if (sigstack):
            old_handles, old_labels = ax.get_legend_handles_labels()
            ax.cla()
            x_comb = x_nobkg+x_nosig
            #x.axis('process').sorting = 'integral'
            the_order = []
            for lab in old_labels:
                for ident in x_comb.identifiers('process'):
                    if (lab == ident.label):
                        the_order.insert(0,ident)
                        break
            if (all_bkg+all_sig>0.): hist.plot1d(x_comb,
                    overlay='process',ax=ax,
                    clear=False,
                    stack=True,
                    order=the_order,
                    fill_opts=fill_opts,
                    error_opts=err_opts,
                    overflow=overflow
                    )

        x_data = x['data']
        all_data = 0
        for key,val in x_data.values().items():
            all_data +=val.sum()
        if (all_data>0 and not blind):
            hist.plot1d(x_data,ax=ax,
                    overlay='process',
                    clear=False,
                    #line_opts=line_opts,
                    error_opts=err_opts_data,
                    overflow=overflow)

            x_allbkg = x_nosig.sum("process",overflow=overflow)
            x_nobkg.scale({p: 1./(lumifb*float(sigscale)) for p in x_nobkg.identifiers('process')}, axis="process")
            x_allsig = x_nobkg.sum("process",overflow=overflow)
            x_data = x_data.sum("process",overflow=overflow)
            x_data.label = 'Data/MC'
            if (not noratio):
                hist.plotratio(x_data,x_allbkg,ax=axr,
                    unc='num',
                    clear=False,
                    error_opts=err_opts_data,
                    denom_fill_opts=err_opts_denom,
                    guide_opts={'linestyle':'dashed','linewidth':1.5},
                    overflow=overflow)
                axr.set_ylim(0. if axr.get_ylim()[0] < 0. else None, 2. if axr.get_ylim()[1] > 2. else None)
            if (dosigrat):
                x_intsosqrtb = x_allsig.copy(content=False)
                sosqrtb = {}
                sosqrtb[var_name] = np.array(x_allsig.axis(var_name).centers())
                sosqrtb["weight"] = np.nan_to_num(np.divide(np.array([sum(x_allsig.values()[()][i:]) for i in range(len(x_allsig.values()[()]))]),np.sqrt([sum(x_allbkg.values()[()][i:]) for i in range(len(x_allbkg.values()[()]))])),0.)
                x_intsosqrtb.fill(**sosqrtb)
                x_intsosqrtb.label = r'$S/\sqrt{B}$'
                hist.plot1d(x_intsosqrtb,ax=axs,
                    clear=False,
                    line_opts=line_opts,
                    overflow=overflow)
                axs.set_ylim(0. if axs.get_ylim()[0] < 0. else None, None)
                axs.get_legend().set_visible(False)
    
    
        print('MC: %.4f Sig: %.4f S/sqrt(B): %.4f - Data: %.4f'%(all_bkg,all_sig,all_sig/math.sqrt(all_bkg),all_data))

    ax.autoscale(axis='x', tight=True)
    if len(xlimits)==2:
        try:
            ax.set_xlim(float(xlimits[0]), None)
        except:
            pass
        try:
            ax.set_xlim(None, float(xlimits[1]))
        except:
            pass
    ax.set_ylim(0, None)
    if (not noratio and dosigrat): 
        ax.xaxis.set_label_text('')
        axr.xaxis.set_label_text('')
        axs.xaxis.set_label_text(var_label)
    elif (not noratio and not dosigrat):
        ax.xaxis.set_label_text('')
        axr.xaxis.set_label_text(var_label)
    elif (noratio and dosigrat):
        ax.xaxis.set_label_text('')
        axs.xaxis.set_label_text(var_label)
    else:
        x.xaxis.set_label_text(var_label)
    ax.ticklabel_format(axis='x', style='sci')
    old_handles, old_labels = ax.get_legend_handles_labels()
    new_labels = []
    for xl in old_labels:
        if ('ggH' in xl and sigscale!=1): xl = xl + " (x " + str(sigscale) + ")"
        new_labels.append(xl)
    leg = ax.legend(handles=old_handles,labels=new_labels,title=r'%s'%plottitle,frameon=True,framealpha=1.0,facecolor='white',ncol=(2 if len(x.identifiers('process')) > 4 else 1))
    lumi = ax.text(1., 1., r"%.1f fb$^{-1}$ (13 TeV)"%lumifb,fontsize=16,horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
    cmstext = ax.text(0., 1., "CMS",fontsize=19,horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes, fontweight='bold')
    if (not solo):
        if (all_data > 0 and not blind): 
            addtext = ax.text(0.085, 1., "Preliminary",fontsize=16,horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes, style='italic')
        else: 
            addtext = ax.text(0.085, 1., "Simulation Preliminary",fontsize=16,horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes, style='italic')
    else: 
        addtext = ax.text(0.085, 1., "Simulation Preliminary",fontsize=16,horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes, style='italic')
    #hep.cms.cmslabel(ax, data=False, paper=False, year='2017')
    fig.savefig("%s_%s_%s_%s_lumi%i.pdf"%(('solo' if solo else 'stack'),sel,var_name,savename,lumifb))
    print("%s_%s_%s_%s_lumi%i.pdf"%(('solo' if solo else 'stack'),sel,var_name,savename,lumifb))
    ax.semilogy()
    minvals = []
    for xd in x.values():
        if not np.trim_zeros(x.values()[xd]).any(): 
            continue
        minvals.append(min(np.trim_zeros(x.values()[xd])))
    if not minvals:
      decsplit = ['0','0']
    else:
      decsplit = str(min(minvals)).split('.')
    if (int(decsplit[0])==0):
        logmin = 0.1**float(len(decsplit[1])-len(decsplit[1].lstrip('0'))+1)
    else:
        logmin = 10.**float(len(decsplit[0])+0)
    ax.set_ylim(logmin/10. if logmin>1. else 0.1, ax.get_ylim()[1]*10.)
    fig.savefig("%s_%s_%s_%s_lumi%i_logy.pdf"%(('solo' if solo else 'stack'),sel,var_name,savename,lumifb))
    print("%s_%s_%s_%s_lumi%i_logy.pdf"%(('solo' if solo else 'stack'),sel,var_name,savename,lumifb))

def getPlots(args):
    print(args.lumi)
    lumifb = float(args.lumi)
    tag = args.tag
    savename = args.savetag

    odir = 'plots/%s/'%tag
    os.system('mkdir -p %s'%odir)
    pwd = os.getcwd()

    hists_mapped = {}
    for h in args.hists:
        # open hists
        hists_unmapped = load('%s.coffea'%h)
        # map to hists
        for key, val in hists_unmapped.items():
            if isinstance(val, hist.Hist):
                if key in hists_mapped:
                    hists_mapped[key] = hists_mapped[key] + processmap.apply(val)
                else:
                    hists_mapped[key] = processmap.apply(val)

    os.chdir(odir)
    # properties
    hist_name = args.hist
    var_name = args.var
    var_label = r"%s"%args.varlabel
    vars_cut =  {}
    #print(args.sel)
    if (len(args.sel)%3==0):
      for vi in range(int(len(args.sel)/3)):
        if (args.sel[vi*3+1]=='neginf'):
          vars_cut[args.sel[vi*3]] = [None, float(args.sel[vi*3+2])]
        elif (args.sel[vi*3+2]=='inf'):
          vars_cut[args.sel[vi*3]] = [float(args.sel[vi*3+1]), None]
        else:
          vars_cut[args.sel[vi*3]] = [float(args.sel[vi*3+1]), float(args.sel[vi*3+2])]
    print(vars_cut)
    sig_cut =  {}
    if (len(args.sigsel)%3==0):
      for vi in range(int(len(args.sigsel)/3)):
        if (args.sigsel[vi*3+1]=='neginf'):
          sig_cut[args.sigsel[vi*3]] = [None, float(args.sigsel[vi*3+2])]
        elif (args.sigsel[vi*3+2]=='inf'):
          sig_cut[args.sigsel[vi*3]] = [float(args.sigsel[vi*3+1]), None]
        else:
          sig_cut[args.sigsel[vi*3]] = [float(args.sigsel[vi*3+1]), float(args.sigsel[vi*3+2])]
    print(sig_cut)
    h = hists_mapped[hist_name]
    print(h)

    drawStack(h,args.hist,var_name,var_label,args.title,lumifb,vars_cut,sig_cut,args.regions,savename,args.xlimits,args.blind,args.solo,args.sigscale,args.sigstack,args.noratio,args.dosigrat,args.overflow)

    os.chdir(pwd)

if __name__ == "__main__":

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

    #ex. python plot_solo.py --hists htt_test --tag test --var jet_pt --varlabel 'p_{T}(jet)' --title Test --lumi 41.5 --sel lep_pt 20. 200. --regions hadel_signal  --hist trigeff --savetag leppt_20
    parser = argparse.ArgumentParser()
    parser.add_argument('--hists',      dest='hists',    default="hists",      help="hists pickle name", nargs='+')
    parser.add_argument('--tag',        dest='tag',      default="",           help="tag")
    parser.add_argument('--savetag',    dest='savetag',  default="",           help="savetag")
    parser.add_argument('--var',        dest='var',      default="",           help="var")
    parser.add_argument('--varlabel',   dest='varlabel', default="",           help="varlabel")
    parser.add_argument('--title',      dest='title',    default="",           help="title")
    parser.add_argument('--lumi',       dest='lumi',     default=50.,          help="lumi",       type=float)
    parser.add_argument('--sel',        dest='sel',      default='',           help='selection',  nargs='+')
    parser.add_argument('--sigsel',     dest='sigsel',   default='',           help='signal selection',  nargs='+')
    parser.add_argument('--regions',    dest='regions',  default='',           help='regionsel',  nargs='+')
    parser.add_argument('--hist',       dest='hist',     default='',           help='histname')
    parser.add_argument('--xlimits',    dest='xlimits',  default='',           help='xlimits',    nargs='+')
    parser.add_argument('--blind',      dest='blind',    default='',           help='blind',      nargs='+')
    parser.add_argument('--solo',       dest='solo',     action='store_true',  help='solo')
    parser.add_argument('--sigscale',   dest='sigscale', default=50,           help='sigscale',   type=int)
    parser.add_argument('--sigstack',   dest='sigstack', action='store_true',  help='sigstack')
    parser.add_argument('--noratio',    dest='noratio',  action='store_true',  help='noratio')
    parser.add_argument('--dosigrat',   dest='dosigrat', action='store_true',  help='dosigrat')
    parser.add_argument('--overflow',   dest='overflow', default='none',       help='overflow')
    args = parser.parse_args()

    getPlots(args)
