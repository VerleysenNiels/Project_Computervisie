ROOMS = { "zaal_a" : ["zaal_b", "zaal_ii"],
          "zaal_b" : ["zaal_a", "zaal_c", "zaal_d", "zaal_e"],
          "zaal_c" : ["zaal_b", "zaal_d"],
          "zaal_d" : ["zaal_b", "zaal_c", "zaal_e", "zaal_h", "zaal_g"],
          "zaal_e" : ["zaal_b", "zaal_d", "zaal_g", "zaal_ii"],
          "zaal_f" : ["zaal_g", "zaal_i", "zaal_ii"],
          "zaal_g" : ["zaal_d", "zaal_e", "zaal_f", "zaal_h", "zaal_i"],
          "zaal_h" : ["zaal_d", "zaal_g", "zaal_m"],
          "zaal_i" : ["zaal_g", "zaal_f", "zaal_j"],
          "zaal_j" : ["zaal_i", "zaal_k"],
          "zaal_k" : ["zaal_j", "zaal_l"],
          "zaal_l" : ["zaal_k", "zaal_iii", "zaal_iv", "zaal_12"],
          "zaal_m" : ["zaal_h", "zaal_n", "zaal_p", "zaal_q"],
          "zaal_n" : ["zaal_m", "zaal_o", "zaal_p"],
          "zaal_o" : ["zaal_n", "zaal_p"],
          "zaal_p" : ["zaal_m", "zaal_n", "zaal_o", "zaal_q", "zaal_r", "zaal_s"],
          "zaal_q" : ["zaal_m", "zaal_p", "zaal_r", "zaal_s"],
          "zaal_r" : ["zaal_p", "zaal_q", "zaal_s"],
          "zaal_s" : ["zaal_p", "zaal_q", "zaal_r", "zaal_iv", "zaal_v"],
          "zaal_ii" : ["zaal_a", "zaal_e", "zaal_f", "zaal_iii", "zaal_1", "zaal_5", "zaal_6"],
          "zaal_iii" : ["zaal_l", "zaal_ii", "zaal_iv", "zaal_12"],
          "zaal_iv" : ["zaal_l", "zaal_s", "zaal_iii", "zaal_v", "zaal_12", "zaal_19"],
          "zaal_v" : ["zaal_s", "zaal_iv", "zaal_19"],
          "zaal_1" : ["zaal_ii", "zaal_2"],
          "zaal_2" : ["zaal_1", "zaal_3", "zaal_4", "zaal_5"],
          "zaal_3" : ["zaal_2", "zaal_4"],
          "zaal_4" : ["zaal_2", "zaal_3", "zaal_5", "zaal_7", "zaal_8"],
          "zaal_5" : ["zaal_ii", "zaal_2", "zaal_4", "zaal_7"],
          "zaal_6" : ["zaal_ii", "zaal_7", "zaal_9"],
          "zaal_7" : ["zaal_4", "zaal_5", "zaal_6", "zaal_8", "zaal_9"],
          "zaal_8" : ["zaal_4", "zaal_7", "zaal_13"],
          "zaal_9" : ["zaal_6", "zaal_7", "zaal_10"],
          "zaal_10" : ["zaal_9", "zaal_11"],
          "zaal_11" : ["zaal_10", "zaal_12"],
          "zaal_12" : ["zaal_l", "zaal_iii", "zaal_iv", "zaal_11"],
          "zaal_13" : ["zaal_8", "zaal_14", "zaal_16", "zaal_17"],
          "zaal_14" : ["zaal_13", "zaal_15", "zaal_16"],
          "zaal_15" : ["zaal_14", "zaal_16"],
          "zaal_16" : ["zaal_13", "zaal_14", "zaal_15", "zaal_17", "zaal_18", "zaal_19"],
          "zaal_17" : ["zaal_13", "zaal_16", "zaal_18", "zaal_19"],
          "zaal_18" : ["zaal_16", "zaal_17", "zaal_19"],
          "zaal_19" : ["zaal_iv", "zaal_v", "zaal_16", "zaal_17", "zaal_19"],
        }

def transition_possible(start, goal):
    """Check if starting point and goal are neighbours or neighbours of neighbours (in this case a room was skipped)"""
    # Check if direct neighbour, this is the most likely transition
    for neighbour in ROOMS.get(start):
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
