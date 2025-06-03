from pydantic_ai import Agent
from typing import List, Tuple
from oaklib import get_adapter
import re


# Define the agent
ols_agent = Agent(
    "openai:gpt-4o",
    system_prompt="""
    You are an expert ontology curator. Use your knowledge of ontologies and of 
    OBO Foundry ontologies to find the best ontology term for a given term supplied by 
    the user. 
    """
)


@ols_agent.tool_plain()
async def find_best_ontology(this_term: str) -> List[Tuple[str, str]]:
    ontology_info = {
        "mondo": "Mondo Disease Ontology",
        "doid": "Human Disease Ontology",
        "hp": "Human Phenotype Ontology",
        "mp": "Mammalian Phenotype Ontology",
        "go": "Gene Ontology",
        "uberon": "Uberon multi-species anatomy ontology",
        "cl": "Cell Ontology",
        "chebi": "Chemical Entities of Biological Interest",
        "pr": "Protein Ontology",
        "so": "Sequence Ontology",
        "pato": "Phenotype And Trait Ontology",
        "maxo": "Medical Action Ontology",
    }

    system_prompt = """
You are an expert in biomedical ontologies. Given a biomedical term, rank the most relevant ontologies
from this list for searching that term. Prioritize domain fit, popularity, and clarity of scope.
Respond as a ranked list of (ontology_id, ontology_title) pairs.
"""

    prompt = f"""
TERM: "{this_term}"

Candidate ontologies:
{chr(10).join(f"- {k}: {v}" for k, v in ontology_info.items())}

Which are the best ontologies in which to search for this term? Output as many as you 
think are relevant, ranked in descending order of relevance.
"""

    response = await ols_agent.call(prompt=prompt, system_prompt=system_prompt)

    ranked = []
    for line in response.splitlines():
        match = re.match(r"\d+\.\s*(\w+)\s*[-–]\s*(.+)", line.strip())
        if match:
            ranked.append((match.group(1), match.group(2)))
    return ranked


async def find_best_curie(term: str) -> List[Tuple[str, str]]:
    ranked_ontologies = await find_best_ontology(term)  # ✅ JUST THIS
    if not ranked_ontologies:
        raise ValueError(f"No suitable ontology found for term: {term}")

    best_ontology_id = ranked_ontologies[0][0]

    adapter = get_adapter("ols:" + best_ontology_id)
    results = adapter.basic_search(term)
    labels = list(adapter.labels(results))
    print(f"## Query: {term} in {best_ontology_id} -> {labels}")
    return labels


import asyncio

if __name__ == "__main__":
    term = "heart disease"
    results = asyncio.run(find_best_curie(term))
    print(f"Results for '{term}':")
    for curie, label in results:
        print(f"{curie}: {label}")
