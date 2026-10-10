"""Make a minimal request to the Gemini API to verify API connectivity."""

import os
import sys

from dotenv import load_dotenv
from google import genai


def main() -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "replace_with_a_new_api_key":
        print(
            "Set GEMINI_API_KEY in .env to a newly generated Google AI Studio key.",
            file=sys.stderr,
        )
        return 1

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with exactly: Gemini API connection successful.",
    )

    if not response.text:
        print("The API request completed but returned no text.", file=sys.stderr)
        return 1

    print("Gemini API connection successful.")
    print(f"Model response: {response.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
