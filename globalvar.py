break_flag = None

def init():
    global break_flag
    break_flag = False

def set_value(flag):
    global break_flag
    break_flag = flag

def get_value():
    return break_flag