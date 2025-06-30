import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/emeka/ros2_drone_ws/install/my_py_drone_pkg'
