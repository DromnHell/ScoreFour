
def get_lines(grid, x, y, z, grid_size) -> list:
    colX, rowX, rowY = [], [], []
    for j in range(grid_size):
        colX.append(grid[x][y][j])
        rowX.append(grid[x][j][z])
        rowY.append(grid[j][y][z])
    return colX, rowX, rowY

def get_diagonals(grid, x, y, z, grid_size) -> list:
    diag1X, diag2X, diag1Y, diag2Y, diag1Z, diag2Z, diag1, diag2, diag3, diag4 = [], [], [], [], [], [], [], [], [], []
    for i in range(y, grid_size):
        for j in range(z, grid_size):
            if i - y == j - z:
                diag1X.append(grid[x][i][j])
        for j in range(z - 1, -1, -1):
            if i - y == z - j:
                diag2X.append(grid[x][i][j])
    for i in range(x, grid_size):
        for j in range(z, grid_size):
            if i - x == j - z:
                diag1Y.append(grid[i][y][j])
        for j in range(z - 1, -1, -1):
            if i - x == z - j:
                diag2Y.append(grid[i][y][j])
        for j in range(y, grid_size):
            if i - x == j - y:
                diag1Z.append(grid[i][j][z])
            for k in range(z, grid_size):
                if i - x == j - y == k - z:
                    print("HERE")
                    diag1.append(grid[i][j][k])
            for k in range(z, -1, -1):
                if i - x == j - y == z - k:
                    diag2.append(grid[i][j][k])
        for j in range(y - 1, -1, -1):
            if i - x == y - j:
                diag2Z.append(grid[i][j][z])
        for j in range(y, -1, -1):
            for k in range(z, grid_size):
                if i - x == y - j == k - z:
                    diag3.append(grid[i][j][k])
            for k in range(z, -1, -1):
                if i - x == y - j == z - k:
                    diag4.append(grid[i][j][k])
    for i in range(y - 1, -1, -1):
        for j in range(z - 1, -1, -1):
            if i - y == j - z:
                diag1X.insert(0, grid[x][i][j])
    for i in range(y, -1, -1):
        for j in range(z, grid_size):
            if i - y == z - j:
                diag2X.insert(0, grid[x][i][j])
    for i in range(x - 1, -1, -1):
            for j in range(z - 1, -1, -1):
                if i - x == j - z:
                    diag1Y.insert(0, grid[i][y][j])
            for j in range(y - 1, -1, -1):
                if i - x == j - y:
                    diag1Z.insert(0, grid[i][j][z])
                for k in range(z - 1, -1, -1):
                    if i - x == j - y == k - z:
                        diag1.insert(0, grid[i][j][k])
                for k in range(z, grid_size):
                    if x - i == y - j == k - z:
                        diag2.insert(0, grid[i][j][k])
            for j in range(y, grid_size):
                for k in range(z - 1, -1, -1):
                    if x - i == j - y == z - k:
                        diag3.insert(0, grid[i][j][k])
                for k in range(z, grid_size):
                    if x - i == j - y == k - z:
                        diag4.insert(0, grid[i][j][k])
    for i in range(x, -1, -1):
        for j in range(z, grid_size):
            if i - x == z - j:
                diag2Y.insert(0,grid[i][y][j])
        for j in range(y, grid_size):
            if i - x == y - j:
                diag2Z.insert(0, grid[i][j][z])
    return diag1X, diag2X, diag1Y, diag2Y, diag1Z, diag2Z, diag1, diag2, diag3, diag4


grid = [[[None, None, None, None], [None, "N", "B", "D"], [None, "R", "Z", "K"], [None, "R", "G", "J"]],
        [["I", None, "N", None], [None, "S", "U", "B"], ["E", "R", "I", "C"], [None, "P", "T", "S"]], 
        [[None, None, None, None], [None, "P", "A", "Z"], [None, "R", "G", "L"], [None, "V", "T", "M"]], 
        [["U", None, "X", None], [None, None, None, None], ["I", None, "O", None], [None, None, None, None]]]

#print(grid[0][1][1])
#print(grid[1][2][2])
#print(grid[2][3][3])

lines = get_lines(grid, 1, 2, 2, 4)
diagonals = get_diagonals(grid, 1, 2, 2, 4)

print(lines)
print(diagonals)

#['A', 'C', 'A', 'B']
#['B', 'A', 'P']
#['G', 'A', 'E']
#['R', 'A', 'L']