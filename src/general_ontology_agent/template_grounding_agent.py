import click
from pathlib import Path
from pydantic_ai import Agent
from typing import List, Tuple
from oaklib import get_adapter


# Define different agents
def create_data_curator_agent():
    return Agent(
        'openai:gpt-4o',
        system_prompt="""
        You are an expert curator of scientific knowledge. Your purpose is to take 
        unstructured scientific text and output structured scientific knowledge that is 
        aligned to a schema that describes the knowledge the user wants to extract.  

        You will be given some scientific text and a schema in LinkML format, and you 
        will output the knowledge contained in the scientific text such that it aligns
        with the LinkML schema. 

        You can output as much or as little data as you think is sensible, as long as it is
        supported by the scientific text. 

        The LinkML schema describes the knowledge that the user wants to extract. Pay particular
        attention to entity types and relationships defined in the schema. These describe
        the types of things the user is interested in, and relationships between them.

        The schema may include some advice about what annotators to use when using the 
        search_ontology tool to ground the terms to the schema. For example, the following items
        in the schema mean that you should use the Mondo Disease Ontology to ground disease 
        terms:

        id_prefixes:
          - MONDO
        annotations:
          annotators: sqlite:obo:mondo

        and the following means that you should use the Human Phenotype Ontology: 

        id_prefixes:
        - HP
        annotations:
          annotators: sqlite:obo:hp

        Some other guidelines:
        1. Use the schema to guide your extraction of knowledge from the scientific text.
        2. Do not respond conversationally, but rather output the structured knowledge without
        any additional commentary.
        """,
        tools=[search_ontology]
    )


def extract(text: str, template: Path):
    data_curator_agent = create_data_curator_agent()
    template_path = template.resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template file {template_path} does not exist.")
    template_text = template_path.read_text()

    response = data_curator_agent.run_sync(
        user_prompt=f"Align this {text} to the schema at {template_text}",
    )
    print(f"Extracted Response: {response}")


@click.command()
@click.option("--agent", default="extract",
              help="Specify the agent to use (e.g., extract, make_schema)")
@click.argument("text", default="This is a statement about diabetes and Marfan syndrome", type=str)
@click.argument("template",
                default="src/general_ontology_agent/linkml_templates/mondo_simple.yaml",
                type=Path)
def main(agent: str, text: str, template: Path):
    """
    Main function to run scientific data curation agent

    Args:
        agent: The name of the agent to use.
        text: The scientific text to align to the schema.
        template: The path to the LinkML schema template.
    """
    if agent == "extract":
        extract(text, template)
    else:
        click.echo(f"Unknown agent: {agent}. Please choose from: extract, make_schema")


async def search_ontology(term: str, ontology: str, n: int, verbose: bool = False) -> List[Tuple[str, str]]:
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
    if verbose:
        print(f"## TOOL USE: Searched for '{term}' in '{ontology}' ontology")
        print(f"## RESULTS: {labels}")
    return labels


if __name__ == "__main__":
    main()
