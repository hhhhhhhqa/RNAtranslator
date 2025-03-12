import os
import subprocess
import pandas as pd
from Bio import SeqIO
from matplotlib import pyplot as plt
import seaborn as sns


IDENTITY_THRESHOLD = 70.0          # Lower percent identity threshold
ALIGNMENT_LENGTH_THRESHOLD = 15     # Lower minimum alignment length
E_VALUE_THRESHOLD = 0.1            # Less stringent E-value threshold
QUERY_COVERAGE_THRESHOLD = 50.0 


query_base_dir = "/data6/helya/RLLM/results/validation/test356200"
target_base_dir = "/data6/helya/dataset/CLIPdb_cluster/cd_hit_results_RBPs/identity_90"
output_base_dir = "./"
generRNA_dir = "/data6/sobhan/GenerRNA"


WORD_SIZE = 7

novelty_results = []

def classify(row, query_lengths):
    query_length = query_lengths.get(row["Query_ID"], 0)
    # Check for zero or missing query length
    if query_length == 0:
        return "Novel"
    
    query_coverage = (row["Alignment Length"] / query_length) * 100
    if (row["% Identity"] < IDENTITY_THRESHOLD and
        row["Alignment Length"] < ALIGNMENT_LENGTH_THRESHOLD and
        row["E-value"] > E_VALUE_THRESHOLD and
        query_coverage < QUERY_COVERAGE_THRESHOLD):
        return "Novel"
    return "Known"

protein_name = "RBM5"

query_file = os.path.join("/data6/sobhan/RLLM/results/validation/RBM5_Pool14/RNAtranslator_ _rnas.fasta")
target_file = os.path.join(target_base_dir, f"{protein_name}_rnas_cdhit_90.fa")
output_file = os.path.join(output_base_dir, f"{protein_name}_blast_output.txt")


if os.path.exists(query_file) and os.path.exists(target_file):
    total_rnas = sum(1 for _ in SeqIO.parse(query_file, "fasta"))
    print(f"Protein: {protein_name} | Total RNAs (from FASTA): {total_rnas}")
    
    blast_command = [
        "blastn",
        "-query", query_file,
        "-subject", target_file,
        "-out", output_file,
        "-perc_identity", str(IDENTITY_THRESHOLD),
        "-evalue", str(E_VALUE_THRESHOLD),
        "-word_size", str(WORD_SIZE),
        "-outfmt", "6 qseqid sseqid pident length evalue"
    ]
    
    print(f"Running BLAST for protein: {protein_name}")
    result = subprocess.run(blast_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running BLAST for {protein_name}:")
        print(result.stderr)

    if os.path.getsize(output_file) == 0:
        print(f"No BLAST hits found for protein {protein_name}. All sequences are considered novel.")
        novelty_results.append({
            "Protein": protein_name,
            "Total RNAs": total_rnas,
            "Known RNAs": 0,
            "Novel RNAs": total_rnas,
            "Known Percentage": 0.0,
            "Novelty Score": 100.0
        })
    else:
        columns = ["Query_ID", "Subject_ID", "% Identity", "Alignment Length", "E-value"]
        df = pd.read_csv(output_file, sep="\t", names=columns)
        
        # Convert columns to proper numeric types
        df["% Identity"] = pd.to_numeric(df["% Identity"], errors='coerce')
        df["Alignment Length"] = pd.to_numeric(df["Alignment Length"], errors='coerce')
        df["E-value"] = pd.to_numeric(df["E-value"], errors='coerce')
        
        # Sort and group to take the best hit per query
        best_hits = df.sort_values(by=["E-value", "% Identity", "Alignment Length"],
                                    ascending=[True, False, False])
        best_hits = best_hits.groupby("Query_ID").first().reset_index()
        
        # Classify each best hit
        query_lengths = {record.id: len(record.seq) for record in SeqIO.parse(query_file, "fasta") if len(record.seq) > 0}


        best_hits["Classification"] = best_hits.apply(lambda row: classify(row, query_lengths), axis=1)

        print(best_hits)

        
        # Build classification dictionary (for reference)
        classification = dict(zip(best_hits["Query_ID"], best_hits["Classification"]))
        print(f"Classification for protein {protein_name}:")
        print(classification)
        
        # Count known RNAs (from BLAST hits)
        known_rnas_from_hits = sum(1 for status in classification.values() if status == "Known")
        # Count RNAs with BLAST hits (unique queries)
        rnas_with_hits = len(classification)
        # RNAs with no BLAST hit are considered novel by default
        no_hit_rnas = total_rnas - rnas_with_hits
        # Total novel RNAs: those classified as Novel from BLAST plus those with no hits
        novel_rnas = (rnas_with_hits - known_rnas_from_hits) + no_hit_rnas
        
        known_percentage = (known_rnas_from_hits / total_rnas) * 100 if total_rnas > 0 else 0.0
        novelty_score = 100.0 - known_percentage
        
        print(f"Protein: {protein_name} | Total RNAs: {total_rnas} | Known RNAs: {known_rnas_from_hits} | Novel RNAs: {novel_rnas}")
        print(f"Known Percentage: {round(known_percentage, 2)}% | Novelty Score: {round(novelty_score, 2)}%")
        
        novelty_results.append({
            "Protein": protein_name,
            "Total RNAs": total_rnas,
            "Known RNAs": known_rnas_from_hits,
            "Novel RNAs": novel_rnas,
            "Known Percentage": round(known_percentage, 2),
            "Novelty Score": round(novelty_score, 2)
        }) 


# novelty_df = pd.DataFrame(novelty_results)
# novelty_output_file = os.path.join(output_base_dir, "novelty_scores.csv")
# novelty_df.to_csv(novelty_output_file, index=False)
# print(f"Novelty scores saved to {novelty_output_file}")


# novel_rna_output_file = os.path.join(generRNA_dir, "novel_rnas_GenerRNA.fasta")
# novel_sequences = []

# if 'classification' not in locals():
#     classification = {}


# for record in SeqIO.parse(query_file, "fasta"):
#     if record.id not in classification or classification[record.id] == "Novel":
#         novel_sequences.append(record)

# SeqIO.write(novel_sequences, novel_rna_output_file, "fasta")
# print(f"Novel RNAs saved to {novel_rna_output_file}")