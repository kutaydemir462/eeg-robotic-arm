import pandas as pd
import datetime

# 1. Load the labels from CSV
df = pd.read_csv("subject-80.csv")

# 2. Process the Marker file (.vmrk)
with open("Bedri331.vmrk", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_marker_lines = []
csv_idx = 0

for line in lines:
    if line.startswith("Mk") and not line.startswith("Mk1="):
        # Format: Mk<n>=Type,Description,Position,Size,Channel
        parts = line.split("=")
        mk_id = parts[0]
        params = parts[1].split(",")

        if csv_idx < len(df):
            row = df.iloc[csv_idx]

            # Update Type and Description
            params[0] = "Stimulus"
            # Replace commas with \1 (BrainVision escape) to prevent parsing errors
            params[1] = row['uyaran_yazi'].replace(",", "\\1")

            # Update Size (Duration) based on CSV timestamps
            start = pd.to_datetime(row['uyaran_baslangic'])
            end = pd.to_datetime(row['uyaran_bitis'])
            duration_s = (end - start).total_seconds()
            params[3] = str(int(duration_s * 500))  # 500Hz sampling rate

            new_line = f"{mk_id}={','.join(params)}"
            new_marker_lines.append(new_line)
            csv_idx += 1
        else:
            new_marker_lines.append(line)
    else:
        new_marker_lines.append(line)

with open("Bedri331_labeled.vmrk", "w", encoding="utf-8") as f:
    f.writelines(new_marker_lines)

# 3. Update the Header file (.vhdr) to point to the new markers
with open("Bedri331.vhdr", "r", encoding="utf-8") as f:
    vhdr_lines = f.readlines()

with open("Bedri331_labeled.vhdr", "w", encoding="utf-8") as f:
    for line in vhdr_lines:
        if line.startswith("MarkerFile="):
            f.write("MarkerFile=Bedri331_labeled.vmrk\n")
        else:
            f.write(line)

print(f"Successfully processed {csv_idx} action timestamps into EEG markers.")