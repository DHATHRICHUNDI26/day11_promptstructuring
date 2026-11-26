
import os
import csv
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-3.5-flash-lite"


class ExtractedInvoice(BaseModel):
    invoice_number: str
    customer_name: str
    invoice_date: str
    total_amount: float


# Load versioned prompt template
with open("prompts_v1.txt", "r", encoding="utf-8") as file:
    prompt_template = file.read()


# Five test inputs
test_inputs = [
    """Invoice Number: INV201
Customer Name: Arjun
Invoice Date: 10-11-2026
Total Amount: 12000""",

    """Invoice Number: INV202
Customer Name: Sneha
Invoice Date: 12-11-2026
Total Amount: 9500""",

    """Invoice Number: INV203
Customer Name: Kiran
Invoice Date: 15-11-2026
Total Amount: 18000""",

    """Invoice Number: INV204
Customer Name: Ananya
Total Amount: 7500""",

    """Invoice Number: INV205
Customer Name: Rohan
Invoice Date: 22-11-2026
Total Amount: 11000"""
]


results = []

for index, invoice_text in enumerate(test_inputs, start=1):

    prompt = prompt_template.replace(
        "{invoice_text}", invoice_text
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ExtractedInvoice
        }
    )

    invoice = ExtractedInvoice.model_validate_json(
        response.text
    )

    output = invoice.model_dump_json()

    print(f"\n--- TEST {index} ---")
    print(output)

    results.append({
        "test_number": index,
        "input": invoice_text,
        "output": output
    })


# Save results in CSV
with open(
    "prompt_test_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["test_number", "input", "output"]
    )

    writer.writeheader()
    writer.writerows(results)


print("\nAll five tests completed.")
print("Results saved in prompt_test_results.csv")
