import pandas as pd

file_path = "dataset/chronic_kidney_disease.arff"

# Read the ARFF file
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Find the @data section
data_start = 0
columns = []

for i, line in enumerate(lines):
    line = line.strip()

    if line.lower().startswith("@attribute"):
        parts = line.split()
        columns.append(parts[1])

    if line.lower() == "@data":
        data_start = i + 1
        break

# Read the data rows
rows = []

for line in lines[data_start:]:
    line = line.strip()

    if line:
        values = [value.strip() for value in line.split(",")]

        # Remove the extra empty value at the end
        if len(values) > len(columns):
            values = values[:len(columns)]

        rows.append(values)

# Create DataFrame
data = pd.DataFrame(rows, columns=columns)

# Save as CSV
data.to_csv("dataset/ckd.csv", index=False)

print("CSV created successfully!")
print("Rows and columns:", data.shape)
print(data.head())