import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"       #Log Format

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str, 
    handlers=[
        logging.FileHandler(log_filepath),  #Save the Log in File
        logging.StreamHandler(sys.stdout)   #Print the Log in Terminal
    ]
)

logger = logging.getLogger("AnimalVision-DirectML Logs")