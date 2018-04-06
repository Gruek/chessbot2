import chess

class Node():
    def __init__(self, score, children=[]):
        self.visits = 0
        self.score = score
        self.child_links = {}
        for child in children:
            self.child_links[child['uci']] = NodeLink(child['weight'])

    def children(self):
        child_nodes = []
        for link in self.child_links.values():
            if link.node:
                child_nodes.append(link.node)
        return child_nodes

class NodeLink():
    def __init__(self, weight):
        self.weight = weight
        self.node = None
        