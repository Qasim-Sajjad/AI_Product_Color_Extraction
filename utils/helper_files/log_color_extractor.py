import pandas as pd
import re

def extract_row_number(line):
    match = re.search(r'Row (\d+):', line)
    if match:
        return int(match.group(1))
    return None

def clean_color(color):
    # Remove asterisks, extra whitespace, and periods
    color = re.sub(r'\*+', '', color).strip().rstrip('.')
    return color

def is_valid_color(color, valid_colors):
    return color.title() in valid_colors

def extract_color_from_text(text, valid_colors):
    # First try direct color match
    color_match = re.search(r'Successfully extracted color: \**([\w\s]+)\**', text)
    if color_match:
        color = clean_color(color_match.group(1))
        if is_valid_color(color, valid_colors):
            return color

    # Try to find color in the text (case insensitive)
    text_lower = text.lower()
    for color in valid_colors:
        if color.lower() in text_lower:
            return color

    return None

def process_log_file(log_file_path):
    # List of valid colors
    valid_colors = {
        'White', 'Gray', 'Red', 'Blue', 'Green', 'Yellow', 'Pink', 'Purple',
        'Orange', 'Brown', 'Beige', 'Gold', 'Silver', 'Navy', 'Teal', 'Maroon',
        'Coral', 'Peach', 'Turquoise', 'Lavender', 'Mint', 'Burgundy', 'Cream',
        'Ivory', 'Black'
    }

    results = []
    processed_rows = set()
    max_row_number = 0
    current_text = ""
    current_row = None
    
    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Check if this is a new log entry
            if line.startswith('2025-'):
                # Process previous entry if exists
                if current_row is not None and 'Successfully extracted color' in current_text:
                    color = extract_color_from_text(current_text, valid_colors)
                    if color:
                        results.append({'row_number': current_row, 'color': color})
                        processed_rows.add(current_row)
                    else:
                        print(f"Could not extract color for row {current_row}")
                        print(f"Text: {current_text}")
                
                # Start new entry
                current_text = line
                current_row = extract_row_number(line)
                if current_row is not None:
                    max_row_number = max(max_row_number, current_row)
            else:
                # Continue current entry
                current_text += line

        # Process the last entry
        if current_row is not None and 'Successfully extracted color' in current_text:
            color = extract_color_from_text(current_text, valid_colors)
            if color:
                results.append({'row_number': current_row, 'color': color})
                processed_rows.add(current_row)
            else:
                print(f"Could not extract color for row {current_row}")
                print(f"Text: {current_text}")

    # Create DataFrame and sort by row number
    df = pd.DataFrame(results)
    df = df.sort_values('row_number').reset_index(drop=True)
    
    # Find missing row numbers
    all_rows = set(range(1, max_row_number + 1))
    missing_rows = sorted(all_rows - processed_rows)
    
    # Save to CSV
    output_file = 'extracted_colors.csv'
    df.to_csv(output_file, index=False)
    print(f"Processed {len(df)} color entries")
    print(f"Results saved to {output_file}")
    
    # Display first few rows
    print("\nFirst few entries:")
    print(df.head())
    
    # Display color distribution
    print("\nColor distribution:")
    print(df['color'].value_counts())
    
    # Save missing rows to file
    missing_rows_file = 'missing_rows.txt'
    with open(missing_rows_file, 'w') as f:
        f.write(f"Total missing rows: {len(missing_rows)}\n\n")
        f.write("Missing row numbers:\n")
        for row in missing_rows:
            f.write(f"{row}\n")
    print(f"\nMissing rows have been saved to {missing_rows_file}")

if __name__ == "__main__":
    log_file_path = "color_extraction_20250415_005328.log"  # Update this path if needed
    process_log_file(log_file_path) 