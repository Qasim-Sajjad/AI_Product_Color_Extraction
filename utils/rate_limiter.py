import time

class RateLimiter:
    def __init__(self, requests_per_min, tokens_per_min):
        self.requests_per_min = requests_per_min
        self.tokens_per_min = tokens_per_min
        self.request_timestamps = []
        self.token_usage = []
        
    def can_make_request(self, estimated_tokens):
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean up old timestamps
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        self.token_usage = [tokens for ts, tokens in self.token_usage if ts > minute_ago]
        
        # Check rate limits
        if (len(self.request_timestamps) >= self.requests_per_min or
            sum(self.token_usage) + estimated_tokens > self.tokens_per_min):
            return False
        
        return True
    
    def record_request(self, tokens_used):
        current_time = time.time()
        self.request_timestamps.append(current_time)
        self.token_usage.append((current_time, tokens_used))
