class Neo4jConfig:
    def __init__(self, uri: str, username: str, password: str):
        """
        Neo4j Configuration
        """
        self.uri = uri
        self.username = username
        self.password = password