import time
from collections import defaultdict
from typing import Dict, Tuple

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)
        
    def check_rate_limit(self, token: str) -> bool:
        """Check if request is within rate limit"""
        current_time = time.time()
        
        # Remove old requests
        self.requests[token] = [
            req_time for req_time in self.requests[token]
            if current_time - req_time < 60
        ]
        
        # Check if under limit
        if len(self.requests[token]) >= self.requests_per_minute:
            return False
            
        # Add new request
        self.requests[token].append(current_time)
        return True
