"""
Main entry point of the program
"""

# - Read the configuration
# - Use asyncIO to read the file from time to time
#   (every 10s... but how do you check it does not drift? could have fixed windows...)
#   (other possibility is to fixed 10s like 0, 10, 20, 30... and just do a kind of "modulo" to group)
# - Each time your read it, compute the statistics and status alert
# - Print these information on the screen

# TODO
