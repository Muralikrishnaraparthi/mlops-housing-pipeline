import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import os

WATCH_DIR = "data/new_data"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class RetrainHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".csv"):
            logging.info(f"New file detected: {event.src_path}")
            try:
                logging.info("Starting retraining...")
                subprocess.run(["python", "train.py"], check=True)
                logging.info("Retraining completed.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Retraining failed: {e}")

if __name__ == "__main__":
    os.makedirs(WATCH_DIR, exist_ok=True)
    event_handler = RetrainHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    logging.info(f"Watching directory: {WATCH_DIR} for new data...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping retraining watcher.")
        observer.stop()
    observer.join()
