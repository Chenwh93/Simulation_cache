break_flag = None
continue_flag = None
round_start_flag = None
ready_flag = None
learn_complete_flag = None

def init():
    global break_flag
    global continue_flag
    global round_start_flag
    global ready_flag
    global learn_complete_flag
    break_flag = False
    continue_flag = False
    round_start_flag = False
    ready_flag = False
    learn_complete_flag = False

def set_break_flag(flag):
    global break_flag
    break_flag = flag

def get_break_flag():
    return break_flag

def set_continue_flag(flag):
    global continue_flag
    continue_flag = flag

def get_continue_flag():
    return continue_flag

def set_round_start_flag(flag):
    global round_start_flag
    round_start_flag = flag

def get_round_start_flag():
    return round_start_flag

def set_ready_flag(flag):
    global ready_flag
    ready_flag = flag

def get_ready_flag():
    return ready_flag

def set_learn_complete_flag(flag):
    global learn_complete_flag
    learn_complete_flag = flag

def get_learn_complete_flag():
    return learn_complete_flag