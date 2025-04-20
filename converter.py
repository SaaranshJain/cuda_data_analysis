import json
import csv

input_json_file = 'data/Electronics_5.json'  # your .jsonl file
output_csv_file = 'output.csv'

# Fields to extract (flattening the nested 'style' field manually if needed)
fields_to_extract = ['overall', 'reviewerID', 'asin',
                     'reviewerName', 'reviewText']

with open(input_json_file, 'r', encoding='utf-8') as json_file, \
     open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:

    writer = None
    for line in json_file:
        if not line.strip():
            continue  # skip blank lines

        record = json.loads(line)

        # Flatten the 'style' field if it exists
        if 'style' in record and isinstance(record['style'], dict):
            # Get first key-value from style dict, e.g., {"Format:": "Hardcover"}
            for k, v in record['style'].items():
                record['style'] = f"{k.strip()} {v.strip()}"
                break  # use only the first item

        # Initialize writer with headers on first row
        if writer is None:
            writer = csv.DictWriter(csv_file, fieldnames=fields_to_extract)
            writer.writeheader()

        # Extract only the fields we care about
        filtered_record = {key: record.get(key, "") for key in fields_to_extract}
        writer.writerow(filtered_record)

print("✅ JSON Lines to CSV conversion done.")
