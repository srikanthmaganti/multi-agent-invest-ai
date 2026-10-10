"""Deterministic output checks and fallback response handling."""

import logging
import re
from typing import List

logger = logging.getLogger("LBG-InvestAI")


class ComplianceGuardrailEngine:
    """
    Prototype phrase checks for draft responses.

    This deterministic check is a limited guardrail; it does not establish
    regulatory compliance.
    """

    PROHIBITED_ADVICE_PATTERNS = [
        r"\byou must invest in\b",
        r"\bguaranteed returns?\b",
        r"\bdefinitely buy\b",
        r"\bi recommend you purchase\b",
        r"\b100% safe\b",
        r"\bcannot lose money\b",
    ]

    REQUIRED_DISCLAIMER = (
        "Note: This information is for educational guidance only and does not constitute "
        "regulated financial advice. Pension value can go down as well as up."
    )

    def evaluate_response(self, text: str) -> tuple[bool, List[str], str]:
        flags = [
            f"BREACH_DEFINITIVE_ADVICE: Triggered pattern '{pattern}'"
            for pattern in self.PROHIBITED_ADVICE_PATTERNS
            if re.search(pattern, text, re.IGNORECASE)
        ]

        is_compliant = not flags
        if flags:
            logger.warning("Compliance violation detected: %s", flags)
            final_text = (
                "Based on FCA regulatory guidelines, I can provide general pension information, "
                "but I cannot recommend specific investment actions. Increasing pension contributions "
                "generally offers tax relief benefits, but you should consider your overall budget. "
                f"\n\n{self.REQUIRED_DISCLAIMER}"
            )
        elif self.REQUIRED_DISCLAIMER not in text:
            final_text = f"{text}\n\n{self.REQUIRED_DISCLAIMER}"
        else:
            final_text = text

        return is_compliant, flags, final_text
