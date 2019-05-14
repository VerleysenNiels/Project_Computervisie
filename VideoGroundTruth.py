import csv

class VideoGroundTruth:

    def __init__(self):
        self.index = 0   # Index to framenumber when next transition happens
        self.frames = [] # Frame numbers when transitions happen
        self.rooms = []  # Previous room at each transition

    def read_file(self, file):
        """Read a file with the transitions between rooms and at which frame in the video they happen"""
        self.clear()

        with open(file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    self.frames.append(int(row["frames"]))
                    self.frames.append(int(row["rooms"]))
                line_count += 1
            print('Processed {line_count} lines from ' + file)

    def room_in_frame(self, frame):
        """Returns the room at a given timestep (frame-number)
           REQUIRES: The file with transitions for the correct video has to be read already
        """
        if frame >= self.frames[self.index]:
            if self.index >= len(self.rooms):
                return None
            else:
                self.index += 1
        return self.rooms[self.index

    def clear(self):
        """Clears the class"""
        self.index = 0
        self.frames.clear()
        self.rooms.clear()
