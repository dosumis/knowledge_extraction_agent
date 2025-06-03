from pydantic_ai import Agent
from typing import List, Tuple
from oaklib import get_adapter


oak_agent = Agent(
    'openai:gpt-4o',
    system_prompt="""
    You are an expert ontology curator. Use the ontologies at your disposal to
    answer the users questions.
    """,
)

@oak_agent.tool_plain
async def search_uberon(term: str) -> List[Tuple[str, str]]:
    """
    Search the UBERON ontology for a term.

    Note that search should take into account synonyms, but synonyms may be incomplete,
    so if you cannot find a concept of interest, try searching using related or synonymous
    terms.

    If you are searching for a composite term, try searching on the sub-terms to get a sense
    of the terminology used in the ontology.

    Args:
        term: The term to search for.

    Returns:
        A list of tuples, each containing an UBERON ID and a label.
    """
    adapter = get_adapter("ols:uberon")
    results = adapter.basic_search(term)
    labels = list(adapter.labels(results))
    print(f"## Query: {term} -> {labels}")
    return labels
