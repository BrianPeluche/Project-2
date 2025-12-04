class NearestNeighbor:
    def __init__(self):
        self.data_x = None
        self.data_y = None

    def datastore(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def test(self, test_point):
        distances = np.linalg.norm(self.data_x - test_point, axis=1)
        nearest_index = np.argmin(distances)
        return self.data_y[nearest_index]
