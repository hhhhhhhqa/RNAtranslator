#!/usr/bin/env python
import os
from src.utils.helpers import fasta_to_dict, read_deepclip_output
from src.utils.plots import (plot_violin_compare, plot_box_compare, plot_ridge_compare,
                             plot_density_compare,)
from src.utils.validations import compare_rna_length, compare_gc_content, compare_mfe_distribution, compare_dG_unfolding_distribution

def evaluate(args):
    rnas = fasta_to_dict(args.rnas_fasta)
    eval_dir = args.eval_dir
    
    if args.deepclip:
        deepclip_json = os.path.join(eval_dir, f"{protein}.json")
        scores = read_deepclip_output(deepclip_json)
        labels = list(scores.keys())
        data = list(scores.values())
        deepclip_outdir = os.path.join(eval_dir, "deepclip")
        os.makedirs(deepclip_outdir, exist_ok=True)
        plot_violin_compare(data, labels, "Binding Score", os.path.join(deepclip_outdir, "binding_affinities_violin.png"))
        plot_box_compare(data, labels, "Binding Score", os.path.join(deepclip_outdir, "binding_affinities_box.png"))
        plot_ridge_compare(data, labels, "Binding Score", os.path.join(deepclip_outdir, "ridge_plot_output.png"))
        plot_density_compare(data, labels, "Binding Score", os.path.join(deepclip_outdir, "density_plot_output.png"))
    
    compare_rna_length(rnas, eval_dir)
    compare_gc_content(rnas, eval_dir)
    compare_mfe_distribution(rnas, eval_dir)
    compare_dG_unfolding_distribution(rnas, eval_dir)
