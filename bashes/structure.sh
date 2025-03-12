#!/bin/bash
# pipeline.sh

# Set the main output directory and the protein file path.
MAIN_OUTPUT_DIR="./structures"
PROTEIN_FILE="/data6/sobhan/RLLM/results/7pcv.pdb"

# Create the main output directory if it doesn't exist.
mkdir -p "$MAIN_OUTPUT_DIR"

# Define the RNA sequences.
declare -A RNA_SEQUENCES
RNA_SEQUENCES["GenerRNA"]="GGAGGUGGUAGGAGGGCCUCCUGGGGACGAGGGGUGGCCCUCGG"
# RNA_SEQUENCES["Random"]="ACGCUCGACGUACGGCAUCGCGAGCGAUUUAC"
# RNA_SEQUENCES["CLIP"]="GGAGGUGGUAGGAGGGCCUCCUGGGGACGAGGGGUGGCCCUCGG"

# Function to run the RosettaRNA prediction and folding.
run_rosetta_rna() {
  local name="$1"
  local fasta_file="$2"
  local output_pdb="$3"

  # Set the directory containing the fasta file and npz file path.
  local rna_dir
  rna_dir=$(dirname "$fasta_file")
  local npz_file="${rna_dir}/${name}.npz"

  echo "Running RosettaRNA prediction for $name..."
  # Run the prediction step.
  python /data6/sobhan/rosseta/trRosettaRNA_v1.1/predict.py -i "$fasta_file" -o "$npz_file"
  if [ $? -ne 0 ]; then
    echo "Error running predict.py for $name" >&2
    exit 1
  fi

  # Run the folding step.
  python /data6/sobhan/rosseta/trRosettaRNA_v1.1/fold.py -out "$output_pdb" -npz "$npz_file" -fa "$fasta_file"
  if [ $? -ne 0 ]; then
    echo "Error running fold.py for $name" >&2
    exit 1
  fi

  echo "Structure prediction done for $name"
}

# Function to run the HDOCKlite docking.
# This function always uses "./Hdock.out" as the output file.
run_hdocklite() {
  local rna_pdb="$1"
  local protein_file="$2"
  
  echo "Running HDOCKlite for RNA PDB: $rna_pdb and Protein: $protein_file"
  
  # Run HDOCK docking with the fixed output file "./Hdock.out".
  /data6/sobhan/docking/HDOCK/hdock "$protein_file" "$rna_pdb" -out ./Hdock.out
  if [ $? -ne 0 ]; then
    echo "Error running hdock" >&2
    exit 1
  fi
  
  # Run the createpl command always using "./Hdock.out".
  /data6/sobhan/docking/HDOCK/createpl ./Hdock.out top100.pdb -nmax 100 -complex -models
  if [ $? -ne 0 ]; then
    echo "Error running createpl" >&2
    exit 1
  fi

  echo "HDOCKlite docking complete for $rna_pdb"
}

# Main pipeline loop.
for name in "${!RNA_SEQUENCES[@]}"; do
  # Create a directory for this RNA.
  RNA_DIR="${MAIN_OUTPUT_DIR}/${name}"
  mkdir -p "$RNA_DIR"
  
  # Write the FASTA file.
  FASTA_FILE="${RNA_DIR}/${name}.fasta"
  {
    echo ">${name}"
    echo "${RNA_SEQUENCES[$name]}"
  } > "$FASTA_FILE"

  # Set the expected output PDB file.
  PREDICTED_PDB="${RNA_DIR}/${name}.pdb"
  echo "Predicting structure for $name (output will be in $RNA_DIR)..."
  
  # Run the RosettaRNA prediction and folding.
  run_rosetta_rna "$name" "$FASTA_FILE" "$PREDICTED_PDB"

  # Change directory into the RNA folder so that the fixed filename ./Hdock.out is valid.
  pushd "$RNA_DIR" > /dev/null
  run_hdocklite "$PREDICTED_PDB" "$PROTEIN_FILE"
  popd > /dev/null
done

echo "Pipeline execution complete."
