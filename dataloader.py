class Dataloader:
    def __init__(self):
        self.data = []
        pass
    
    def load_data(self, source):
        # Simulate loading data from a source
        self.data = [1, 2, 3, 4, 5]
        print(f"Data loaded from {source}: {self.data}")
    
    def get_data(self):
        return self.data