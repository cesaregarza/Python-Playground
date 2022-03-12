# %%
import networkx as nx

# %%
def generate_latin_square_graph(n:int) -> nx.Graph:
    """Generates an nxn graph that will contain the connections for a Latin Square

    Args:
        n (int): size of latin square

    Returns:
        nx.Graph: Graph containing nodes and connections necessary
    """
    g = nx.Graph()
    #Add all nxn nodes
    for row in range(n):
        for col in range(n):
            g.add_node((row, col))
    
    #Generate all connections between nodes
    for row in range(n):
        for col in range(n):
            #Previous rows and columns have a connection already. This reduces duplicate work.
            row_neighbors = [row_ for row_ in range(row, n) if row_ != row]
            col_neighbors = [col_ for col_ in range(col, n) if col_ != col]
            for row_neighbor in row_neighbors:
                g.add_edge((row, col), (row_neighbor, col))
            for col_neighbor in col_neighbors:
                g.add_edge((row, col), (row, col_neighbor))
    return g

# %%
k = 9
g = generate_latin_square_graph(9)
nx.draw(g, with_labels=True)

# %%
def decision_coloring(graph:nx.Graph, num_colors:int, pre_solved:dict[tuple[int,int],int] = {}) -> dict[tuple[int,int], int]:
    """Color the input graph according to the number of colors given

    Args:
        graph (nx.Graph): Graph to color
        num_colors (int): Number of colors to use

    Returns:
        dict[tuple[int,int], int]: Dictionary of node colors
    """
    #Initialize the color map. If a pre-solved dictionary is given, accept them as true.
    color_map   = {**pre_solved}
    graph_order = int(len(graph.nodes) ** 0.5)
    
    def int_index_to_tuple(idx:int) -> tuple[int,int]:
        return (idx // graph_order, idx % graph_order)
    
    def safe_to_color(tup:tuple[int,int], color:int) -> bool:
        neighbors       = graph.neighbors(tup)
        neighbor_colors = [color_map.get(neighbors, None) for neighbors in neighbors]
        return color not in neighbor_colors
    
    def backtrack_solver(int_index:int) -> bool:
        """Recursive backtracking solver for coloring the graph

        Args:
            int_index (int): Index of the node to color

        Returns:
            bool: True if the graph has been colored, None if the graph failed to color
        """
        #Convert the integer index to the (row, col) form
        idx = int_index_to_tuple(int_index)

        #If the integer index has reached the end of the graph, return True.
        #This will only happen once the graph has been colored.
        if int_index == len(graph.nodes):
            return True
        
        #Iterate through all possible colors, ignoring neighbors.
        for color in range(num_colors):
            #If the color is safe to use, assign it to the node and recurse with the next node.
            if safe_to_color(idx, color):
                color_map[idx] = color
                #Because the recursion will only return True if every subsequent node has been colored correctly,
                #the graph is colored if the recursion returns True.
                if backtrack_solver(int_index + 1):
                    return True
                #If the recursion fails, remove the previous assignment and try again
                color_map.pop(idx)
    
    if backtrack_solver(0) is None:
        raise ValueError("Could not solve graph")
    
    return color_map

# %%
def generate_latin_square(n:int) -> list[list[int]]:
    """Generates a latin square of size n

    Args:
        n (int): size of latin square

    Returns:
        list[list[int]]: List of lists representing latin square
    """
    g           = generate_latin_square_graph(n)
    color_map   = decision_coloring(g, n)
    return [[color_map[(row, col)] for col in range(n)] for row in range(n)]

# %%
generate_latin_square(11)

# %%
def validate_latin_square(square:list[list[int]]) -> bool:
    """Validates that the input is a valid latin square

    Args:
        square (list[list[int]]): List of lists representing latin square

    Returns:
        bool: True if valid, False otherwise
    """
    n = len(square)
    for row in range(n):
        for col in range(n):
            cell = square[row][col]
            for i in range(n):
                if (square[row][i] == cell) and (i != col):
                    print(f"Row {row} has duplicate {cell} at {col} and {i}")
                    return False
                if (square[i][col] == cell) and (i != row):
                    print(f"Column {col} has duplicate {cell} at {row} and {i}")
                    return False
    return True

validate_latin_square(generate_latin_square(11))

# %%



