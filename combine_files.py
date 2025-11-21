import csv

# Read existing quiz_data.csv
data = []
with open('quiz_data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # Get header
    data.append(header)
    for row in reader:
        data.append(row)

print(f'Read {len(data)-1} entries from quiz_data.csv')

# Read new_entries.csv
new_count = 0
with open('new_entries.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
        new_count += 1

print(f'Read {new_count} entries from new_entries.csv')

# Write combined data back to quiz_data.csv
with open('quiz_data.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(f'Successfully combined files!')
print(f'Total entries in quiz_data.csv: {len(data)-1} (original + {new_count} new)')




