import csv
import logging


class RoomGraph:
    """ Class for inference on room transitions given painting predictions. """

    def __init__(self, file):
        self.rooms = {}
        self.read_file(file)

        # Necessary lists for algorithm
        self.current_path = []
        self.path_indices = []
        self.detected_rooms = []
        self.detected_counters = []

    def read_file(self, file):
        """ Read a file with the neighours for every room. """
        self.rooms.clear()

        with open(file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                self.rooms[row["room"]] = row["neighbours"].split('-')
                line_count += 1

            logging.info('Processed %d lines from %s', line_count, file)
            logging.debug(self.rooms)

    def transition_possible(self, start, goal):
        """ Check if starting point and goal are neighbours or neighbours of 
            neighbours (in this case a room was skipped) 
        """
        # Check if direct neighbour, this is the most likely transition
        if self.rooms.get(start):
            for neighbour in self.rooms.get(start):
                if neighbour == goal:
                    return True
        # Still not found -> most likely not a real transition
        return False

    def highest_likely_path(self, guessed_room, confidence):
        """ Determine the most likely followed path from the list of rooms and
            corresponding counters.
        """

        # Update counter:
        index = self.update_counter_room(guessed_room, confidence)

        # Case 1: guessed room is last room from path
        if index == len(self.current_path) - 1:
            return False, self.current_path

        # Case 2: new room can be added to previous path
        if len(self.current_path) == 0 or self.transition_possible(self.current_path[-1], guessed_room):
            self.current_path.append(guessed_room)
            self.path_indices.append(index)
            return True, self.current_path

        # Check if a segment on the end of the current path should be changed
        path_index = len(self.current_path) - 2
        room_score = self.detected_counters[index]
        segment_score = self.detected_counters[self.path_indices[path_index]]

        finished = False
        while not finished and room_score > segment_score:
            if path_index == -1 or self.transition_possible(guessed_room, self.current_path[path_index - 1]):
                # Case 3: segment on the end of the current path should be replaced by the guessed room

                # Remove segment from path
                self.current_path = self.current_path[:path_index + 1]
                self.path_indices = self.path_indices[:path_index + 1]

                # Add guessed room to path
                self.current_path.append(guessed_room)
                self.path_indices.append(index)

                finished = True

            else:
                # Increase size of path segment
                c = self.path_indices[path_index]
                stop = False
                path_index -= 1
                segment_score += self.detected_counters[self.path_indices[path_index]]

                # Check if guessed room has a counter between previous and new segment-border
                while not stop and c > self.path_indices[path_index]:
                    if self.detected_rooms[c] == guessed_room:
                        room_score += self.detected_counters[c]
                        stop = True
                    c -= 1

        if finished:
            # Case 3
            return True, self.current_path
        else:
            # Case 4: Segment should not be replaced by guessed room
            return False, self.current_path

    def update_counter_room(self, room, confidence):
        """ Update counter of room or add new counter for room.

        Returns:
            int -- The index of the room
        """
        index = len(self.detected_counters) - 1

        # Find position of counter or determine that new counter needs to be added
        found = False
        while index >= 0 and index >= self.path_indices[-1] and not found:
            if self.detected_rooms[index] == room:
                found = True
            else:
                index -= 1

        if found:
            # Update counter
            self.detected_counters[index] += 1  # confidence
        else:
            # Add new counter
            index = len(self.detected_counters)
            self.detected_rooms.append(room)
            self.detected_counters.append(1)  # confidence)

        return index
