# src/main.py
"""
MCP Server for Tax Code Search
Exposes search functionality via Model Context Protocol
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

from embeddings.vectorStore import get_vectorstore
from retrieval.search import search

# Initialize MCP server
app = Server("tax-code-search")

# Load vector store once at startup
print("Loading vector store...", file=sys.stderr)
try:
    vector_store = get_vectorstore("chroma_db")
    print("✓ Vector store loaded successfully", file=sys.stderr)
    
    # Initialize BM25 for hybrid search
    from retrieval.search import initialize_bm25
    initialize_bm25(vector_store)
    print("✓ Hybrid search initialized", file=sys.stderr)
except Exception as e:
    print(f"✗ Error loading vector store: {e}", file=sys.stderr)
    sys.exit(1)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Define available tools for the MCP client"""
    return [
        Tool(
            name="search_tax_code",
            description=(
                "Search the US Tax Code (Title 26 - Internal Revenue Code) for relevant sections. "
                "Returns the top k most relevant passages with section references and page numbers. "
                "Useful for finding information about tax deductions, credits, income rules, and other tax law provisions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'SALT deduction limits', 'senior citizen deductions', 'Section 164')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 20)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from the MCP client"""
    
    if name == "search_tax_code":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        
        # Validate inputs
        if not query:
            return [TextContent(
                type="text",
                text="Error: Query cannot be empty"
            )]
        
        top_k = min(max(1, top_k), 20)  # Clamp between 1 and 20
        
        # Log the search
        print(f"Searching for: '{query}' (top_k={top_k})", file=sys.stderr)
        
        try:
            # Perform search
            results = search(vector_store, query, k=top_k)
            
            if not results:
                return [TextContent(
                    type="text",
                    text=f"No results found for query: '{query}'"
                )]
            
            # Format results
            formatted_output = []
            formatted_output.append(f"Found {len(results)} results for query: '{query}'\n")
            formatted_output.append("=" * 80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                formatted_output.append(f"**Result {i}**\n")
                formatted_output.append(f"Section: {result['section']}\n")
                formatted_output.append(f"Page: {result['page']}\n")
                formatted_output.append(f"Relevance Score: {result['score']:.3f}\n")
                formatted_output.append(f"\nContent:\n{result['text']}\n")
                formatted_output.append("\n" + "-" * 80 + "\n\n")
            
            return [TextContent(
                type="text",
                text="".join(formatted_output)
            )]
            
        except Exception as e:
            print(f"Error during search: {e}", file=sys.stderr)
            return [TextContent(
                type="text",
                text=f"Error performing search: {str(e)}"
            )]
    
    return [TextContent(
        type="text",
        text=f"Unknown tool: {name}"
    )]


async def main():
    """Run the MCP server"""
    print("Starting Tax Code MCP Server...", file=sys.stderr)
    print("Waiting for client connection...", file=sys.stderr)
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down server...", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)