import numpy as np

class Cluster:
    def __init__(self, piece_id, max_length):
        """
        Initializes the cluster with the origin at (0,0).
        :param piece_id: ID of the initial piece.
        :param max_length: maximum allowed length for rows or columns.
        """
        self.piece_id = piece_id
        self.max_length = max_length
        self.cluster_array = {}  # (row, col) -> piece_id
        self.cluster_array[(0, 0)] = piece_id

    def _find_position(self, piece_id):
        """
        Finds the position of a given piece_id.
        Returns (row, col) or None if not found.
        """
        for pos, pid in self.cluster_array.items():
            if pid == piece_id:
                return pos
        return None

    def _can_insert(self, row, col):
        """
        Checks if a piece can be inserted at (row, col).
        """
        if (row, col) in self.cluster_array:
            return False  # Spot already taken

        # Check if adding this piece exceeds max_length
        all_rows = [r for r, _ in self.cluster_array.keys()] + [row]
        all_cols = [c for _, c in self.cluster_array.keys()] + [col]
        if max(all_rows) - min(all_rows) + 1 > self.max_length:
            return False
        if max(all_cols) - min(all_cols) + 1 > self.max_length:
            return False

        return True

    def _insert_at(self, row, col, new_piece_id):
        """
        Inserts piece at (row, col) if possible.
        """
        if self._can_insert(row, col):
            self.cluster_array[(row, col)] = new_piece_id
            return True
        return False

    def insert_relative_origin(self, direction, new_piece_id):
        """
        Insert relative to the origin piece.
        direction: 0=top, 1=right, 2=bottom, 3=left
        """
        origin_pos = self._find_position(self.piece_id)
        if origin_pos is None:
            return False
        if self._find_position(new_piece_id) is not None:
            return False
        row, col = origin_pos
        if direction == 0:
            return self._insert_at(row - 1, col, new_piece_id)
        elif direction == 1:
            return self._insert_at(row, col + 1, new_piece_id)
        elif direction == 2:
            return self._insert_at(row + 1, col, new_piece_id)
        elif direction == 3:
            return self._insert_at(row, col - 1, new_piece_id)
        return False

    def insert_relative_piece(self, target_piece_id, direction, new_piece_id):
        """
        Insert relative to a specific piece.
        """
        pos = self._find_position(target_piece_id)
        if pos is None:
            return False

        if self._find_position(new_piece_id) is not None:
            return False

        row, col = pos
        if direction == 0:
            return self._insert_at(row - 1, col, new_piece_id)
        elif direction == 1:
            return self._insert_at(row, col + 1, new_piece_id)
        elif direction == 2:
            return self._insert_at(row + 1, col, new_piece_id)
        elif direction == 3:
            return self._insert_at(row, col - 1, new_piece_id)
        return False

    def __str__(self):
        """
        Print the cluster as a grid.
        """
        if not self.cluster_array:
            return ""

        rows = [r for r, _ in self.cluster_array.keys()]
        cols = [c for _, c in self.cluster_array.keys()]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        grid = []
        for r in range(min_row, max_row + 1):
            row_list = []
            for c in range(min_col, max_col + 1):
                row_list.append(str(self.cluster_array.get((r, c), '.')))
            grid.append("\t".join(row_list))
        return "\n".join(grid)

def solve_greedy_new(score_matrix, num_pieces):
    length = int(np.ceil(num_pieces ** 0.5))
    clusters = []  # empty list to hold the Cluster objects

    for i in range(num_pieces):
        cluster = Cluster(piece_id=i, max_length=length)
        clusters.append(cluster)


    # Flatten the array with indices
    flat_with_indices = [(i, j, score_matrix[i][j]) for i in range(len(score_matrix)) for j in range(len(score_matrix[0]))]

    # Sort by value
    flat_with_indices.sort(key=lambda x: x[2], reverse=True)

    while True:
        # Loop over sorted values
        for i, j, value in flat_with_indices:
            piece_id_one = i // 4
            piece_id_two = j // 4
            if piece_id_one == piece_id_two: continue

            # 0:Top, 1:Right, 2:Bottom, 3:Left
            piece_type_one = i % 4
            piece_type_two = j % 4
            if piece_type_one == piece_type_two: continue
            if (piece_type_one + piece_type_two) % 2 != 0: continue

            inserted = False
            for cluster in clusters:
                if cluster.piece_id == piece_id_two:
                    continue
                if cluster.piece_id == piece_id_one:
                    inner_inserted = cluster.insert_relative_origin(direction = piece_type_one, new_piece_id = piece_id_two)
                else:
                    inner_inserted = cluster.insert_relative_piece(target_piece_id = piece_id_two, direction=piece_type_two, new_piece_id=piece_id_two)
                if len(cluster.cluster_array) == num_pieces:
                    return cluster
                if not inserted:
                    inserted = inner_inserted

            if inserted:
                break

def solve_greedy_newer(score_matrix, num_pieces):
    length = int(np.ceil(num_pieces ** 0.5))
    cluster = None

    # Flatten the array with indices
    flat_with_indices = [(i, j, score_matrix[i][j]) for i in range(len(score_matrix)) for j in range(len(score_matrix[0]))]

    # Sort by value
    flat_with_indices.sort(key=lambda x: x[2], reverse=True)

    while True:
        # Loop over sorted values
        for i, j, value in flat_with_indices:
            piece_id_one = i // 4
            piece_id_two = j // 4
            if piece_id_one == piece_id_two:
                continue

            # 0:Top, 1:Right, 2:Bottom, 3:Left
            piece_type_one = i % 4
            piece_type_two = j % 4
            if piece_type_one == piece_type_two:
                continue
            if (piece_type_one + piece_type_two) % 2 != 0:
                continue

            if cluster is None:
                cluster = Cluster(piece_id=piece_id_one, max_length=length)

            if cluster.piece_id == piece_id_two:
                continue

            #testing
            if piece_id_one == 6 or piece_id_two == 6:
                x = 5

            if cluster.piece_id == piece_id_one:
                inserted = cluster.insert_relative_origin(direction = piece_type_one, new_piece_id = piece_id_two)
            else:
                inserted = cluster.insert_relative_piece(target_piece_id = piece_id_one, direction=piece_type_one, new_piece_id=piece_id_two)
            if len(cluster.cluster_array) == num_pieces:
                return cluster

            if inserted:
                break

def reconstruct_from_cluster(cluster, pieces_images):
    if not cluster.cluster_array:
        return None

    rows = [r for r, c in cluster.cluster_array.keys()]
    cols = [c for r, c in cluster.cluster_array.keys()]

    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    sample_img = pieces_images[int(next(iter(cluster.cluster_array.values())))]
    h, w, c = sample_img.shape

    total_h = (max_r - min_r + 1) * h
    total_w = (max_c - min_c + 1) * w

    canvas = np.zeros((total_h, total_w, c), dtype=np.uint8)

    for (r, c), pid in cluster.cluster_array.items():
        norm_r = r - min_r
        norm_c = c - min_c

        y = norm_r * h
        x = norm_c * w

        canvas[y:y + h, x:x + w] = pieces_images[int(pid)]

    return canvas