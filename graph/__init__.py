"""GraphRAG subpackage.

Provides entity/relation extraction, Neo4j-backed graph storage,
Leiden community detection, hierarchical community summarization,
graph-local and graph-global retrieval, and an LLM-based query router.
"""

from graph.builder import GraphBuilder
from graph.communities import CommunityDetector, CommunityLevel
from graph.extractor import EntityRelationExtractor, ExtractionResult
from graph.neo4j_client import Neo4jClient, Neo4jUnavailable
from graph.retriever import GraphRetriever
from graph.router import QueryRouter, RouteDecision, RouteMode
from graph.summarizer import CommunitySummarizer

__all__ = [
    "Neo4jClient",
    "Neo4jUnavailable",
    "EntityRelationExtractor",
    "ExtractionResult",
    "GraphBuilder",
    "CommunityDetector",
    "CommunityLevel",
    "CommunitySummarizer",
    "GraphRetriever",
    "QueryRouter",
    "RouteDecision",
    "RouteMode",
]
