import numpy as np

class vector_ring_buffer:
    def __init__(self, capacity, dtype=np.float64):
        self.capacity = capacity
        self.buffer = np.zeros((capacity,3), dtype=dtype)
        self.head = 0  # Next write position
        self.tail = 0  # Oldest data position
        self.size = 0  # Number of elements currently in the buffer

    def append(self, value):
        self.buffer[self.head] = value
        self.head = (self.head + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.tail = (self.tail + 1) % self.capacity

    def get_values(self):
        if self.size == 0:
            return np.array([], dtype=self.buffer.dtype)
        elif self.head > self.tail:
            # Data is contiguous
            return self.buffer[self.tail:self.head]
        else:
            # Data wraps around
            return np.concatenate((self.buffer[self.tail:], self.buffer[:self.head]))