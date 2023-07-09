import measure
import network
import threading
import signal
import sys


if __name__ == '__main__':
    measure_thread = threading.Thread(target=lambda: measure.run())
    measure_thread.daemon = True
    measure_thread.start()
    network.run(sys.argv)