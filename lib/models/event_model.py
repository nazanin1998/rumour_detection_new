class EventModel:
    def __init__(self, path, rumors, non_rumors):
        self.name = path
        self.rumors = rumors
        self.non_rumors = non_rumors

    def to_string(self, index=''):
        return'\t\tEvent ' + str(index) + ' ==> ' + self.name + '\t==>\trumours : ' + str(
            len(self.rumors)) + ',\tnon_rumours : ' + str(
            len(self.non_rumors))
