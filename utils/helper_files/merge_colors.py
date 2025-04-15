import pandas as pd

def merge_colors(products_file, colors_file, output_file):
    # Read the dataframes
    print("Loading data...")
    products_df = pd.read_csv(products_file)
    colors_df = pd.read_csv(colors_file)
    
    # Keep row number same for products list
    colors_df['index'] = colors_df['row_number']
    
    # Create a mapping of index to color
    color_mapping = dict(zip(colors_df['index'], colors_df['color']))
    
    # Update the color column in products_df, preserving existing values
    print("Updating colors...")
    # Only update rows where we have new color information
    for idx, color in color_mapping.items():
        if pd.isna(products_df.at[idx, 'Color (product.metafields.shopify.color-pattern)']):
            products_df.at[idx, 'Color (product.metafields.shopify.color-pattern)'] = color
    
    # Save the updated dataframe
    print(f"Saving updated data to {output_file}...")
    products_df.to_csv(output_file, index=False)
    
    # Print some statistics
    total_products = len(products_df)
    updated_colors = products_df['Color (product.metafields.shopify.color-pattern)'].notna().sum()
    print(f"\nStatistics:")
    print(f"Total products: {total_products}")
    print(f"Products with colors: {updated_colors}")
    print(f"Products without colors: {total_products - updated_colors}")
    
    # Print color distribution
    print("\nColor distribution:")
    print(products_df['Color (product.metafields.shopify.color-pattern)'].value_counts())

if __name__ == "__main__":
    # File paths
    products_file = "products_data/cleaned_products_list_with_colors.csv"
    colors_file = "extracted_colors.csv"
    output_file = "products_data/cleaned_products_list_with_colors_2.csv"
    
    # Run the merge
    merge_colors(products_file, colors_file, output_file) 