import pandas as pd

def merge_colors(cleaned_file, complete_file, output_file):
    # Read both dataframes
    print("Loading data...")
    cleaned_df = pd.read_csv(cleaned_file)
    complete_df = pd.read_csv(complete_file)
    
    # Create a mapping dictionary using Handle and Image Position as composite key
    print("Creating color mapping...")
    color_mapping = {}
    for _, row in cleaned_df.iterrows():
        if pd.notna(row['Handle']) and pd.notna(row['Image Position']):
            key = (row['Handle'], row['Image Position'])
            color_mapping[key] = row['Color (product.metafields.shopify.color-pattern)']
    
    # Update colors in complete_df
    print("Updating colors in complete products list...")
    updated_count = 0
    for idx, row in complete_df.iterrows():
        if pd.notna(row['Handle']) and pd.notna(row['Image Position']):
            key = (row['Handle'], row['Image Position'])
            if key in color_mapping:
                complete_df.at[idx, 'Color (product.metafields.shopify.color-pattern)'] = color_mapping[key]
                updated_count += 1
    
    # Save the updated complete products list
    print(f"Saving updated data to {output_file}...")
    complete_df.to_csv(output_file, index=False)
    
    # Print statistics
    total_rows = len(complete_df)
    print(f"\nStatistics:")
    print(f"Total rows in complete products list: {total_rows}")
    print(f"Rows with updated colors: {updated_count}")
    print(f"Rows without color updates: {total_rows - updated_count}")
    
    # Print color distribution
    print("\nColor distribution in updated file:")
    print(complete_df['Color (product.metafields.shopify.color-pattern)'].value_counts())

if __name__ == "__main__":
    # File paths
    cleaned_file = "products_data/cleaned_products_list_with_colors_2.csv"
    complete_file = "products_data/complete_products_list.csv"
    output_file = "products_data/complete_products_list_with_colors.csv"
    
    # Run the merge
    merge_colors(cleaned_file, complete_file, output_file) 