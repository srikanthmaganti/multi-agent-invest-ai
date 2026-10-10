"""Streamlit chat interface for the Invest AI orchestrator."""

import os

import requests
import streamlit as st


def format_api_error(response: requests.Response) -> str:
    """Convert API JSON errors, including Pydantic field errors, into readable text."""
    try:
        error_payload = response.json()
    except ValueError:
        return response.text or "The API returned an empty error response."

    details = error_payload.get("detail", error_payload)
    if isinstance(details, list):
        formatted = []
        for item in details:
            if isinstance(item, dict):
                location = " -> ".join(str(part) for part in item.get("loc", []))
                message = item.get("msg", "Invalid value")
                formatted.append(f"{location}: {message}" if location else message)
        if formatted:
            return "\n".join(formatted)
    if isinstance(details, str):
        return details
    return str(details)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("**Retrieved FCA references**")
    for source in sources:
        pages = ", ".join(str(page) for page in source.get("page_numbers", []))
        page_label = f"p. {pages}" if pages else "page not available"
        section = source.get("section")
        label = f"{source['document_title']} — {page_label}"
        if section:
            label += f" — {section}"
        st.markdown(f"- [{label}]({source['source_url']})")


st.set_page_config(page_title="Invest AI", page_icon="💬")
st.title("Invest AI")
st.caption(
    "General educational information about pensions and investing. "
    "This is not regulated financial advice."
)

with st.sidebar:
    st.header("Your demo profile")
    customer_id = st.text_input("Anonymized customer ID", value="DEMO-CUSTOMER")
    age_band = st.selectbox(
        "Age band", ["18-24", "25-34", "35-50", "51-65", "65+"], index=2
    )
    investor_experience = st.selectbox(
        "Investing experience", ["Novice", "Intermediate", "Advanced"], index=1
    )
    has_pension = st.checkbox("I have a pension", value=True)

    api_url = os.getenv("INVEST_AI_API_URL", "http://127.0.0.1:8000").rstrip("/")
    st.caption(f"Orchestrator API: {api_url}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        metadata = message.get("metadata")
        if metadata:
            with st.expander("Response details"):
                st.write(f"Compliance check passed: {metadata['is_compliant']}")
                st.write(f"Latency: {metadata['execution_latency_ms']} ms")
                if metadata["compliance_flags"]:
                    st.write("Compliance flags:", metadata["compliance_flags"])
                render_sources(metadata.get("sources", []))

prompt = st.chat_input("Ask about pensions or investing")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "context": {
            "customer_id": customer_id.strip() or "DEMO-CUSTOMER",
            "age_band": age_band,
            "investor_experience": investor_experience,
            "has_pension": has_pension,
        },
        "query_text": prompt,
    }

    with st.chat_message("assistant"):
        with st.spinner("Preparing a response..."):
            try:
                response = requests.post(
                    f"{api_url}/api/chat",
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
            except requests.ConnectionError:
                st.error(
                    "Cannot connect to the orchestrator API. Start it with "
                    "`python run_server.py` and try again."
                )
            except requests.Timeout:
                st.error("The orchestrator API timed out. Please try again.")
            except requests.HTTPError as exc:
                if exc.response is None:
                    st.error(f"The orchestrator API returned an error: {exc}")
                elif exc.response.status_code == 422:
                    st.error(
                        "Please correct the request and try again:\n\n"
                        f"{format_api_error(exc.response)}"
                    )
                elif exc.response.status_code == 503:
                    detail = format_api_error(exc.response)
                    if "Gemini is temporarily unavailable" in detail:
                        st.error(
                            "Gemini is temporarily busy. Please wait a moment and try again."
                        )
                    else:
                        st.error(detail)
                else:
                    st.error(
                        "The orchestrator API returned an error: "
                        f"{format_api_error(exc.response)}"
                    )
            except requests.RequestException as exc:
                st.error(f"Unable to reach the orchestrator API: {exc}")
            else:
                if result.get("status") == "REJECTED":
                    answer = result["message"]
                    metadata = None
                else:
                    answer = result["generated_guidance"]
                    metadata = {
                        "is_compliant": result["is_compliant"],
                        "execution_latency_ms": result["execution_latency_ms"],
                        "compliance_flags": result["compliance_flags"],
                        "sources": result.get("sources", []),
                    }

                st.markdown(answer)
                if metadata:
                    with st.expander("Response details"):
                        st.write(f"Compliance check passed: {metadata['is_compliant']}")
                        st.write(f"Latency: {metadata['execution_latency_ms']} ms")
                        if metadata["compliance_flags"]:
                            st.write("Compliance flags:", metadata["compliance_flags"])
                        render_sources(metadata.get("sources", []))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata,
                    }
                )
