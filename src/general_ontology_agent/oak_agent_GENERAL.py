import click
from pydantic_ai import Agent
from typing import List, Tuple
from oaklib import get_adapter


oak_agent = Agent(
    'openai:gpt-4o',
    system_prompt="""
    You are an expert ontology curator. Your purpose is to search ontologies
    to find the closest ontology terms to user query. 
    
    You can search in any OBO ontology you want, but these are the suggested ontologies
    to use:
    
    ontology_info = {
        "mondo": ("Mondo Disease Ontology", "Use for human disease-related queries"),
        "hp": ("Human Phenotype Ontology", "Use for human phenotype-related queries"),
        "mp": ("Mammalian Phenotype Ontology", "Use for mammalian phenotype-related queries"),
        "go": ("Gene Ontology", "Use for gene function and biological process queries"), 
        "uberon": ("Uberon multi-species anatomy ontology", "Use for anatomical structure queries"),
        "cl": ("Cell Ontology", "Use for cell type and lineage queries"),
        "chebi": ("Chemical Entities of Biological Interest", "Use for chemical and molecular queries"),
        "pr": ("Protein Ontology", "Use for protein-related queries"),
        "so": ("Sequence Ontology", "Use for sequence-related queries"),
        "pato": ("Phenotype And Trait Ontology", "Use for phenotype and trait-related queries"),
        "maxo": ("Medical Action Ontology", "Use for medical action-related queries"),
    }
    
    Some other guidelines:
    1. Search for the user query in the ontology you think is most appropriate.
    2. Only return the ontology ID and label
    3. Only return ontology terms that are very good matches to the query. They should 
    be the same or very close to the user query.
    4. If you don't find a suitable term, try searching using related or synonymous terms.
    The related term should be a term that is CLOSELY RELATED to the original query
    5. DO NOT respond conversationally. Return an ontology term and label that best matches
    the user query.
    """
)


@oak_agent.tool_plain
async def search_ontology(term: str, ontology: str, n: int) -> List[Tuple[str, str]]:
    """
    Search an OBO ontology for a term.

    Note that search should take into account synonyms, but synonyms may be incomplete,
    so if you cannot find a concept of interest, try searching using related or synonymous
    terms.

    If you are searching for a composite term, try searching on the sub-terms to get a sense
    of the terminology used in the ontology.

    Args:
        term: The term to search for.
        ontology: The ontology ID to search
        n: The number of results to return.

    Returns:
        A list of tuples, each containing an ontology ID and a label.
    """
    adapter = get_adapter("ols:" + ontology)
    results = adapter.basic_search(term)
    labels = list(adapter.labels(results))
    print(f"## TOOL USE: Searched for '{term}' in '{ontology}' ontology")
    print(f"## RESULTS: {labels}")
    return labels


@click.command()
@click.argument("query", default="diabetes", type=str)
def main (query: str):
    """
    Main function to run the OAK agent

    Args:
        query: The term to search for in the ontology.
    """

    # Modified to be more explicit about what we want
    response = oak_agent.run_sync(
        user_prompt=f"Find the best matching ontology term for: {query}"
    )
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
