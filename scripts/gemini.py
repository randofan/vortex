import os
import csv
import re
import asyncio
import aiofiles
from google import genai
from google.genai import types


async def process_paintings_csv(csv_path, output_csv="predictions.csv"):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"

    prompt = """Closely examine this Western European painting. Only consider the painting itself. DO NOT USE ANY METADATA. Think carefully about what artistic movement it could be a part of and who the painter could be. Using these two attributes and any additional details about the painting, predict the exact year it was painted.

Format you response as:
# Reasoning: [include reasoning here]
# Prediction: [include exact year prediction here as plain text]"""

    # Read all rows first
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Rate limiter: 2000 requests per minute = ~33.33 requests per second
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
    rate_limit = 25  # requests per second

    async def process_single_image(row, index):
        async with semaphore:
            # Rate limiting: space out requests
            if index > 0:
                await asyncio.sleep(1 / rate_limit)

            image_path = f"/home/davsong/vortex/data/{row['path']}"
            actual_year = row["year"]
            # print(f"Processing {image_path} with actual year {actual_year}")

            try:
                # Read image file asynchronously
                async with aiofiles.open(image_path, "rb") as f:
                    image_bytes = await f.read()

                # Generate content with image and prompt
                contents = [
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ),
                    prompt,
                ]

                # Use async API call
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                )

                # Extract year prediction from response
                response_text = response.text
                prediction, reasoning = extract_prediction_and_reasoning(response_text)

                # print(f"Processed {image_path}: {actual_year} -> {prediction}")
                return prediction, reasoning

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                return "", ""

    # Process all images concurrently
    tasks = [process_single_image(row, i) for i, row in enumerate(rows)]
    predictions = await asyncio.gather(*tasks)

    # Write predictions to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["year", "reasoning"])
        for pred, reason in predictions:
            writer.writerow([pred, reason])

    print(f"Predictions saved to {output_csv}")


def extract_prediction_and_reasoning(response_text):
    """Extract year from the last line and reasoning from all but the last line"""
    lines = response_text.strip().split("\n")

    # Get reasoning (all lines except the last)
    reasoning_lines = lines[:-1]
    reasoning = "\n".join(reasoning_lines).strip().replace("\n", " ")

    # Get prediction from last line
    last_line = lines[-1]
    prediction = ""
    if "Prediction:" in last_line:
        # Extract everything after "Prediction: "
        prediction_part = last_line.split("Prediction:", 1)[-1].strip()
        # Extract just the year (4 digits)
        year_match = re.search(r"\b(\d{4})\b", prediction_part)
        if year_match:
            prediction = year_match.group(1)

    return prediction, reasoning


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
        asyncio.run(process_paintings_csv(csv_path, output_path))
    else:
        print("Usage: python gemini.py <input_csv> [output_csv]")
