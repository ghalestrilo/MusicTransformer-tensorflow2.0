import os
from model import MusicTransformerDecoder
import tensorflow as tf
import numpy as np
import miditoolkit
#import modules
import pickle
import utils
import time
import argparse
import math
import params as par
import pprint

from threading import Thread, Event
from pythonosc import dispatcher, osc_server


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# TODO: Move to engine.py
state = {
    'is_running': False,
    'history': [],
    'temperature': 1.2,
    'tick_interval': 0.25,
    'buffer_length': 8,
    'model': None,
    'debug_output': True
}

# Clock implements the main loop
class Clock(Thread):
    def __init__(self, event):
      Thread.__init__(self)
      self.stopped = event

    def run(self):
      model = state['model']
      while not self.stopped.wait(state['tick_interval']):
        if (state['is_running'] == True):
          # seq = model.tick()
          state['history'][0] = model.tick()[-128:]
          # print('tick: {}'.format(seq))
          # history = list([list(h[-16:]) for h in history])
          if (state['debug_output']):
            print('history: {}'.format([model.decode(h) for h in state['history'][0]]))


def start_timer():
    stopFlag = Event()
    state['stopFlag'] = stopFlag
    state['clock'] = Clock(stopFlag)
    state['clock'].start()


def stop_timer():
    # this will stop the timer
    state['stopFlag'].set()

def engine_set(unused_addr, args):
    try:
        field, value = args
        state[field] = value
        print("[{0}] ~ {1}".format(field, value))
    except KeyError:
        print("no such key ~ {0}".format(field))
        pass


def push_event(unused_addr, event):
    print("[event] ~ {0}".format(event))
    state['history'][0].append(event)


def engine_print(unused_addr, args=None):
    field = args
    if (args == None):
      pprint.pprint(state)
      return
    try:
        # data = [state['model'].word2event[word] for word in state[field][0]] if field == 'history' else state[field]
        data = state[field]
        print("[{0}] ~ {1}".format(field, data))
    except KeyError:
        print("no such key ~ {0}".format(field))
        pass


def sample_model(unused_addr, args):
    model = args[0]
    event = model.predict()
    print(event)


def prepare_model(unused_addr, args):
    model = args[0]
    event = model.realtime_setup(state)
    print(event)


def bind_dispatcher(dispatcher, model):
    state['model'] = model
    dispatcher.map("/start", engine_set, 'is_running', True)
    dispatcher.map("/pause", engine_set, 'is_running', False)
    dispatcher.map("/reset", lambda _: state['history'].clear())
    dispatcher.map("/debug", engine_print)
    dispatcher.map("/event", push_event)  # event2word

    dispatcher.map("/set", lambda addr, k, v: engine_set(addr, [k, v]))

    if (model):
        dispatcher.map("/sample", sample_model, model)
        dispatcher.map("/prep",   prepare_model, model)


def load_model():
    return MusicTransformerDecoder(
        embedding_dim=256, vocab_size=par.vocab_size,
        num_layer=6,
        max_seq=state['max_seq'],
        dropout=0.1,
        debug=False)

# /TODO: Move to engine.py


# Main
if __name__ == "__main__":
    
    # Parse CLI Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq', default=256, help='최대 길이', type=int)
    parser.add_argument('--state', default="0",
                    help='the initial state of the improv', type=str)

    parser.add_argument("--ip",
                        default="127.0.0.1", help="The ip to listen on")
    parser.add_argument("--port",
                        type=int, default=5005, help="The port to listen on")
    args = parser.parse_args()

    state['max_seq'] = args.max_seq
    state['history'] = [[int(x) for x in str(args.state).split(',')]]


    # Prep Model
    model = load_model()
    prepare_model(None, [model])  # for real time use

    # Prep Server
    dispatcher = dispatcher.Dispatcher()
    bind_dispatcher(dispatcher, model)

    # Start Server
    server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    start_timer()
    server.serve_forever()
    stop_timer()
    model.close()
