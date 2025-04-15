import pandas as pd
import base64,os,requests,io,time,tempfile,logging
from PIL import Image
from datetime import datetime
from groq import Groq
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from utils.rate_limiter import RateLimiter
from utils.config import VISION_MODEL, VISION_MODEL2, RATE_LIMITS, TARGET_IMAGE_SIZE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'color_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)

# Load environment variables
load_dotenv()

class GroqProcessor:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.current_model = VISION_MODEL
        
        # Initialize rate limiters for both models
        self.rate_limiters = {
            VISION_MODEL: RateLimiter(
                RATE_LIMITS[VISION_MODEL]['requests_per_min'],
                RATE_LIMITS[VISION_MODEL]['tokens_per_min']
            ),
            VISION_MODEL2: RateLimiter(
                RATE_LIMITS[VISION_MODEL2]['requests_per_min'],
                RATE_LIMITS[VISION_MODEL2]['tokens_per_min']
            )
        }
        
        # Track daily requests for both models
        self.daily_requests = {
            VISION_MODEL: 0,
            VISION_MODEL2: 0
        }
        self.last_reset_day = datetime.now().date()

    def _reset_daily_counts_if_needed(self):
        current_date = datetime.now().date()
        if current_date != self.last_reset_day:
            self.daily_requests = {
                VISION_MODEL: 0,
                VISION_MODEL2: 0
            }
            self.last_reset_day = current_date

    def _can_use_model(self, model):
        self._reset_daily_counts_if_needed()
        return self.daily_requests[model] < RATE_LIMITS[model]['requests_per_day']

    def _select_available_model(self, estimated_tokens):
        try:
            # Check if current model can be used
            current_limiter = self.rate_limiters[self.current_model]
            
            # Try current model first
            if (self._can_use_model(self.current_model)):
                logging.info(f"Using current model: {self.current_model}")
                return self.current_model
            
            # Try alternate model
            alternate_model = VISION_MODEL2 if self.current_model == VISION_MODEL else VISION_MODEL
            alternate_limiter = self.rate_limiters[alternate_model]
            
            if (self._can_use_model(alternate_model)):
                self.current_model = alternate_model
                logging.info(f"Switching to alternate model: {alternate_model}")
                return alternate_model
            
            # If neither model is immediately available, select the one with more daily requests remaining
            vision_remaining = RATE_LIMITS[VISION_MODEL]['requests_per_day'] - self.daily_requests[VISION_MODEL]
            vision2_remaining = RATE_LIMITS[VISION_MODEL2]['requests_per_day'] - self.daily_requests[VISION_MODEL2]
            
            selected_model = VISION_MODEL if vision_remaining > vision2_remaining else VISION_MODEL2
            self.current_model = selected_model
            logging.info(f"Both models rate limited, selecting model with more remaining requests: {selected_model}")
            return selected_model

        except Exception as e:
            logging.error(f"Error in _select_available_model: {str(e)}")
            logging.error(f"Current model: {self.current_model}")
            logging.error(f"Daily requests: {self.daily_requests}")
            # Default to current model if there's an error
            return self.current_model

    def process_image(self, image_url, prompt):
        try:
            logging.info(f"Starting to process image from URL: {image_url}")
            
            # Download and process image
            try:
                response = requests.get(image_url)
                response.raise_for_status()  # Raise an exception for bad status codes
                logging.info(f"Successfully downloaded image from URL: {image_url}")
            except Exception as e:
                logging.error(f"Failed to download image from URL {image_url}: {str(e)}")
                return None

            try:
                img = Image.open(io.BytesIO(response.content))
                logging.info(f"Successfully opened image with size: {img.size}")
            except Exception as e:
                logging.error(f"Failed to open image: {str(e)}")
                return None
            
            # Resize image
            try:
                img = img.resize(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
                logging.info(f"Successfully resized image to {TARGET_IMAGE_SIZE}")
            except Exception as e:
                logging.error(f"Failed to resize image: {str(e)}")
                return None
            
            # Save to temporary file
            try:
                # Convert RGBA to RGB if necessary
                if img.mode == 'RGBA':
                    # Create a white background image
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    # Paste the image on the background using alpha channel
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode != 'RGB':
                    # Convert any other mode to RGB
                    img = img.convert('RGB')

                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    img.save(temp_file, format='JPEG', quality=85)
                    temp_path = temp_file.name
                logging.info(f"Successfully saved temporary file: {temp_path}")
            except Exception as e:
                logging.error(f"Failed to save temporary file: {str(e)}")
                return None

            # Encode image
            try:
                with open(temp_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                logging.info("Successfully encoded image to base64")
            except Exception as e:
                logging.error(f"Failed to encode image: {str(e)}")
                return None
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                    logging.info("Successfully cleaned up temporary file")
                except Exception as e:
                    logging.error(f"Failed to clean up temporary file: {str(e)}")

            estimated_tokens = 200  # Approximate token estimation
            
            while True:
                try:
                    selected_model = self._select_available_model(estimated_tokens)
                    if not selected_model:
                        logging.error("No model available")
                        return None
                    
                    logging.info(f"Selected model for processing: {selected_model}")
                    current_limiter = self.rate_limiters[selected_model]
                    
                    try:
                        logging.info(f"Making API call to model {selected_model}")
                        chat_completion = self.client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}",
                                            },
                                        },
                                    ],
                                }
                            ],
                            model=selected_model,
                        )

                        current_limiter.record_request(estimated_tokens)
                        self.daily_requests[selected_model] += 1
                        
                        result = chat_completion.choices[0].message.content.strip()
                        logging.info(f"Successfully got result from model {selected_model}: {result}")
                        
                        # Add sleep time between requests
                        logging.info("Sleeping for 5 seconds before next request")
                        time.sleep(5)
                        
                        return result
                        
                    except Exception as e:
                        logging.error(f"API call failed for model {selected_model}: {str(e)}")
                        logging.error(f"API call error details: {type(e).__name__}")
                        if selected_model == self.current_model:
                            self.current_model = VISION_MODEL2 if selected_model == VISION_MODEL else VISION_MODEL
                            logging.info(f"Switching to alternate model: {self.current_model}")
                            continue
                        raise

                except Exception as e:
                    logging.error(f"Error in model selection/processing loop: {str(e)}")
                    logging.error(f"Error type: {type(e).__name__}")
                    logging.error(f"Error details: {str(e)}")
                    return None

        except Exception as e:
            logging.error(f"Critical error in process_image: {str(e)}")
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Error details: {str(e)}")
            return None

def main():
    # Load API keys
    api_keys = [
        os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 5)
        if os.getenv(f"GROQ_API_KEY_{i}")
    ]

    # Load CSV
    df = pd.read_csv('products_data/cleaned_products_list_with_colors.csv')
    
    # Extract rows with missing colors
    missing_colors_df = df[df['Color (product.metafields.shopify.color-pattern)'].isna()].copy()
    print(f"Found {len(missing_colors_df)} rows with missing colors")
    
    prompt = """You will be given an image.

Task: Identify the dominant color visible on the cloth worn by the woman in the image.

Instructions:

Use only the following list of colors to answer:
White, Gray, Red, Blue, Black, Green, Yellow, Pink, Purple, Orange, Brown, Beige, Gold, Silver, Navy, Teal, Maroon, Coral, Peach, Turquoise, Lavender, Mint, Burgundy, Cream, Ivory.

Return only one color name from the list. Do not include any explanation, description, or additional text—just the color name."""

    # Create processors for each API key
    vision_processors = [GroqProcessor(key) for key in api_keys]
    
    # Split DataFrame into chunks based on number of API keys
    chunks = np.array_split(missing_colors_df, len(api_keys))
    
    def process_chunk(args):
        chunk_df, processor = None, None
        try:
            logging.info("Starting to unpack chunk arguments")
            chunk_df, processor = args
            logging.info(f"Successfully unpacked chunk with {len(chunk_df)} rows")
            results = []
            
            # Initialize statistics
            success_count = 0
            error_count = 0
            skipped_count = 0
            
            # Create a progress bar for this chunk
            chunk_id = api_keys.index(processor.client.api_key) + 1
            logging.info(f"Processing chunk {chunk_id} with API key ending in ...{processor.client.api_key[-4:]}")
            
            pbar = tqdm(
                total=len(chunk_df),
                desc=f'Processor {chunk_id}',
                position=chunk_id,
                leave=True
            )
            
            for idx, row in chunk_df.iterrows():
                try:
                    logging.info(f"Processing row {idx} in chunk {chunk_id}")
                    
                    image_url = row['Image Src']
                    if pd.isna(image_url):
                        logging.info(f"Row {idx}: No image URL found")
                        results.append((idx, None))
                        skipped_count += 1
                        continue
                    
                    logging.info(f"Row {idx}: Processing image from URL: {image_url}")
                    color = processor.process_image(image_url, prompt)
                    
                    if color is not None:
                        success_count += 1
                        logging.info(f"Row {idx}: Successfully extracted color: {color}")
                        print(f"\rP{chunk_id} | ✓ Row {idx}: {color}", flush=True)
                    else:
                        error_count += 1
                        logging.error(f"Row {idx}: Failed to extract color")
                        print(f"\rP{chunk_id} | ✗ Row {idx}: Failed", flush=True)
                    
                    results.append((idx, color))
                    pbar.set_postfix({'✓': success_count, '✗': error_count, '⭕': skipped_count})
                    pbar.update(1)
                    
                except Exception as e:
                    logging.error(f"Error processing row {idx} in chunk {chunk_id}: {str(e)}")
                    logging.error(f"Exception type: {type(e).__name__}")
                    logging.error(f"Exception traceback: ", exc_info=True)
                    error_count += 1
                    results.append((idx, None))
                    pbar.update(1)
            
            pbar.close()
            stats = {'success': success_count, 'error': error_count, 'skipped': skipped_count}
            logging.info(f"Chunk {chunk_id} processing complete. Success: {success_count}, Errors: {error_count}, Skipped: {skipped_count}")
            return (results, stats)
        
        except Exception as e:
            chunk_id = api_keys.index(processor.client.api_key) + 1 if processor else "Unknown"
            chunk_size = len(chunk_df) if chunk_df is not None else "Unknown"
            logging.error(f"Critical error in process_chunk for chunk {chunk_id} (size: {chunk_size}): {str(e)}")
            logging.error("Full exception details:", exc_info=True)
            # Return partial results if available, otherwise empty results
            if 'results' in locals() and 'success_count' in locals():
                stats = {'success': success_count, 'error': error_count + 1, 'skipped': skipped_count}
                return (results, stats)
            return ([], {'success': 0, 'error': 1, 'skipped': 0})
        finally:
            if 'pbar' in locals():
                try:
                    pbar.close()
                except:
                    pass

    # Create work items - pair each chunk with its dedicated processor
    work_items = list(zip(chunks, vision_processors))
    
    print("\n=== Starting Color Extraction Process ===")
    print(f"Total products to process: {len(missing_colors_df)}")
    print(f"Using {len(api_keys)} API keys in parallel")
    print("Progress bars below show processing status for each API key\n")
    
    # Process chunks in parallel with statistics
    with ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
        all_results = list(executor.map(process_chunk, work_items))

    # Separate results and statistics
    results_list = []
    stats_list = []
    for result in all_results:
        try:
            chunk_results, chunk_stats = result  # Unpack the tuple
            results_list.append(chunk_results)
            stats_list.append(chunk_stats)
        except Exception as e:
            logging.error(f"Error unpacking results: {str(e)}")
            results_list.append([])
            stats_list.append({'success': 0, 'error': 0, 'skipped': 0})
    
    # Flatten results and sort by original index
    flat_results = []
    for chunk_results in results_list:
        flat_results.extend(chunk_results)
    
    # Sort by original index
    flat_results.sort(key=lambda x: x[0])
    
    # Create a new DataFrame with the results, preserving original indices
    results_df = pd.DataFrame(flat_results, columns=['original_index', 'color'])
    results_df.set_index('original_index', inplace=True)
    
    # Update the original DataFrame with new colors, using the original indices
    df.loc[results_df.index, 'Color (product.metafields.shopify.color-pattern)'] = results_df['color']
    
    # Calculate and display total statistics
    total_stats = {
        'success': sum(s['success'] for s in stats_list),
        'error': sum(s['error'] for s in stats_list),
        'skipped': sum(s['skipped'] for s in stats_list)
    }
    
    print("\n=== Processing Complete ===")
    print(f"Successfully processed: {total_stats['success']} products")
    print(f"Errors encountered: {total_stats['error']} products")
    print(f"Skipped (no image URL): {total_stats['skipped']} products")
    print(f"Total processed: {sum(total_stats.values())} products")
    
    # Save results
    output_file = 'products_data/cleaned_products_list_with_colors_2.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    logging.info("Processing completed")

if __name__ == "__main__":
    # Clear terminal for better visibility
    os.system('cls' if os.name == 'nt' else 'clear')
    main()