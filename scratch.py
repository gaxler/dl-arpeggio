from fabric import Connection
import os
IP_ = "34.27.255.59"
conn = Connection(IP_, connect_kwargs={"key_filename": os.path.expanduser('~/.ssh/google_compute_engine'), })  
print("df")