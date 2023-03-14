class Node:
    def __init__(self, state, parent = None, action = None) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.childs = list()

    def add_child(self, child) -> None:
        self.childs.append(child)


class Tree:
    def __init__(self, initial_state) -> None:
        self.root_node = Node(initial_state)
        self.nodes = [self.root_node]

    def add_node(self, node, parent) -> None:
        parent.add_child(node)
        self.nodes.append(node)

    def create_node(self, state, parent = None, action = None) -> Node:
        node = Node(state, parent, action)
        if parent:
            parent.add_child(node)
        self.nodes.append(node)
        return node

    def get_node(self, state) -> Node:
        for node in self.nodes:
            if node.state == state:
                return node
        return None

    def reset_tree(self, initial_state) -> None:
        self.root_node = Node(initial_state)
        self.nodes = [self.root_node]
