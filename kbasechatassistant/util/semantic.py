from typing import Dict, List
from langchain_community.graphs import Neo4jGraph

class Semantic:
    def __init__(self, graph: Neo4jGraph = None):
        self.graph = graph if graph else Neo4jGraph()

    def get_user_id(self) -> int:
        """
        Placeholder for a function that would normally retrieve
        a user's ID
        """
        return 1
    
    
    def remove_lucene_chars(self, text: str) -> str:
        """Remove Lucene special characters"""
        special_chars = [
            "+",
            "-",
            "&",
            "|",
            "!",
            "(",
            ")",
            "{",
            "}",
            "[",
            "]",
            "^",
            '"',
            "~",
            "*",
            "?",
            ":",
            "\\",
        ]
        for char in special_chars:
            if char in text:
                text = text.replace(char, " ")
        return text.strip()
    
    
    def generate_full_text_query(self, input: str) -> str:
        """
        Generate a full-text search query for a given input string.
    
        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~0.8) to each word, then combines them using the AND
        operator. Useful for mapping KBase apps and DataObjects from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query = ""
        words = [el for el in self.remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~0.85 AND"
        full_text_query += f" {words[-1]}~0.85"
        return full_text_query.strip()
    
    
    def get_candidates(self,input: str, type: str, limit: int = 3) -> List[Dict[str, str]]:
        """
        Retrieve a list of candidate entities from database based on the input string.
    
        This function queries the Neo4j database using a full-text search. It takes the
        input string, generates a full-text query, and executes this query against the
        specified index in the database. The function returns a list of candidates
        matching the query, with each candidate being a dictionary containing their name
        and label.
        """
        candidate_query = """
        CALL db.index.fulltext.queryNodes($index, $fulltextQuery, {limit: $limit})
        YIELD node
        RETURN node.name AS candidate
        """
        ft_query = self.generate_full_text_query(input)
        candidates = self.graph.query(
            candidate_query, {"fulltextQuery": ft_query, "index": type, "limit": limit}
        )
        return candidates