import csv

class RoomGraph:

    def __init__(self, file):
        self.rooms = {}
        self.read_file(file)

    def read_file(self, file):
        """Read a file with the neighours for every room"""
        self.rooms.clear()

        with open(file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    self.rooms[row["room"]] = row["neighbours"].split('-')
                line_count += 1
            print('Processed ' + str(line_count) + ' lines from ' + file)
            print(self.rooms)

    def transition_possible(self, start, goal):
        """Check if starting point and goal are neighbours or neighbours of neighbours (in this case a room was skipped)"""
        # Check if direct neighbour, this is the most likely transition
        if self.rooms.get(start):
            for neighbour in self.rooms.get(start):
                if neighbour == goal:
                    return True

        # Goal is not a direct neighbour, check if a room was skipped (for instance by rushing through it)
        # NOTE: currently not used because it causes more errors (there are more possible transitions)
        """for neighbour in ROOMS.get(start):
            for neighbourOfNeighbour in ROOMS.get(neighbour):
                if neighbourOfNeighbour == goal:
                    return True
        """
        # Still not found -> most likely not a real transition
        return False
