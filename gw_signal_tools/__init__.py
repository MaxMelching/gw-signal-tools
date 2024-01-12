
# TODO: decide if functions shall be imported here. Otherwise one has to
# import from each module (preferred solution at the moment)


# ---------- Initialize Logging ----------
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s (%(filename)s: %(lineno)d): %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S'
)

# TODO: handle errors via logging? On the other hand, we log to command line anyway...

# logging.captureWarnings(True)  # Makes formatting a bit worse, can this be changed?
