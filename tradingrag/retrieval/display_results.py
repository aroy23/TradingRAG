import json
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

def display_results(results_file: str = "rag_results.json"):
    """
    Display RAG query results in a nicely formatted way.
    
    Args:
        results_file: Path to the JSON file containing results
    """
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {results_file} not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {results_file}")
        return

    console = Console()
    
    # Check if there's an analysis section
    analysis = None
    if isinstance(data[-1], dict) and "analysis" in data[-1]:
        analysis = data[-1]["analysis"]
        data = data[:-1]  # Remove analysis from results

    # Create a table for the results
    table = Table(title="Query Results", show_header=True, header_style="bold magenta")
    table.add_column("Symbol", style="cyan")
    table.add_column("Period", style="green")
    table.add_column("Return", justify="right", style="yellow")
    table.add_column("Similarity", justify="right", style="blue")
    table.add_column("Description", style="white")

    # Add rows to the table
    for result in data:
        table.add_row(
            result["symbol"],
            f"{result['start_date']} to {result['end_date']}",
            f"{result['return']:.2f}%",
            f"{result['similarity_score']:.2f}",
            result["description"]
        )

    # Display the table
    console.print(table)

    # Display analysis if available
    if analysis:
        console.print("\n[bold]Analysis:[/bold]")
        console.print(Panel(analysis, title="GPT-4 Analysis", border_style="blue"))

def main():
    parser = argparse.ArgumentParser(description="Display RAG query results")
    parser.add_argument("--file", type=str, default="rag_results.json", 
                       help="Path to the results JSON file")
    args = parser.parse_args()
    
    display_results(args.file)

if __name__ == "__main__":
    main() 