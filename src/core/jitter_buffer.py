import heapq
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class BufferedMessage:
    """Wraps a message with its timestamp for priority queue sorting."""
    timestamp: int  # Use nanoseconds from ROS 2
    # topic: str = field(compare=False)
    connection: Any = field(compare=False)
    data: Any = field(compare=False)

class JitterBuffer:
    def __init__(self, delay_ns: int):
        """
        :param delay_ns: How long (in nanoseconds) to hold messages 
                         to account for jitter. (e.g., 100ms = 100,000,000)
        """
        self.buffer = []  # The priority queue (min-heap)
        self.delay_ns = delay_ns
        self.current_time = 0

    def push(self, timestamp: int, connection: Any, data: Any):
        """Add a new message from the rosbag into the buffer."""
        heapq.heappush(self.buffer, BufferedMessage(timestamp, connection, data))
        # Keep track of the 'latest' time we've seen in the bag
        self.current_time = max(self.current_time, timestamp)

    def pop_ready(self):
        """
        Yields messages that have 'aged' past the delay threshold 
        and are now safe to process in chronological order.
        """
        while self.buffer and (self.current_time - self.buffer[0].timestamp) >= self.delay_ns:
            yield heapq.heappop(self.buffer)

    def flush(self):
        """Yields all remaining messages (call this at end of bag)."""
        while self.buffer:
            yield heapq.heappop(self.buffer)