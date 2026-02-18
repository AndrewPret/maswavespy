import os
import glob

data_dir = rf"O:\SysEng SW\Projects_General_Access\RD0077 Sensonic\passive-MASW\MP1_manual_picks_corrected"

# Find all c0 and f0 files
c0_files = glob.glob(rf'{data_dir}\c0_section*.txt')
f0_files = glob.glob(rf'{data_dir}\f0_section*.txt')

# Extract section numbers from filenames
get_section_number = lambda filename: int(filename.split('section')[-1].split('.')[0])

c0_sections = {get_section_number(f): f for f in c0_files}
f0_sections = {get_section_number(f): f for f in f0_files}

# Find sections that exist in both
common_sections = sorted(set(c0_sections.keys()) & set(f0_sections.keys()))

# Loop through common sections
for section in common_sections:
    c0_path = c0_sections[section]
    f0_path = f0_sections[section]
    output_path = rf'{data_dir}\section{section}.dc'

    # Read the data
    with open(f0_path, 'r') as f:
        f0_values = [float(line.strip()) for line in f if line.strip()]
    with open(c0_path, 'r') as f:
        c0_values = [float(line.strip()) for line in f if line.strip()]

    # Check matching lengths
    if len(f0_values) != len(c0_values):
        print(f"Skipping section {section}: mismatched lengths")
        continue

    # Write to output file
    with open(output_path, 'w') as out:
        out.write(f'>>Start {section}\n')
        for f_val, c_val in zip(f0_values, c0_values):
            out.write(f'DATA    {f_val:.2f}  {c_val:.1f}    0.5    0\n')

    print(f"Wrote: {output_path}")
