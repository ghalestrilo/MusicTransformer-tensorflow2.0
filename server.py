import os
import sys

import tensorflow as tf
import numpy as np
import miditoolkit
#import modules
import pickle
import time
import argparse
import math
import pprint

from threading import Thread, Event
from pythonosc import dispatcher, osc_server, udp_client


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# TODO: Move to engine.py
state = {
    'is_running': False,
    'history': [],
    'temperature': 1.2,
    'until_next_event': 0.25,
    'buffer_length': 8,
    'playback': True,
    'model_name': None,
    'model': None,
    'scclient': None,
    'debug_output': True,
}
NOTE_OFFSET=60

# Clock implements the main loop
class Clock(Thread):
    def __init__(self, event):
      Thread.__init__(self)
      self.stopped = event
      self.playhead = 0

    #def should_generate():

    def get_next(self):
      return state['model'].decode(state['history'][0][self.playhead])

    def play_next(self):
      e = self.get_next()
      if (e == None):
        print("No event / history is empty")
        return
      if (e.type == 'note_on'):
        play(int(e.value))
        print('playing {}'.format(e.value))
      if (e.type == 'time_shift'):
        state['until_next_event'] = e.value / 1000
        print('waiting {}'.format(e.value / 1000))


    def run(self):
      model = state['model']
      trigger = 0.75

      while not self.stopped.wait(state['until_next_event']):
        if (state['is_running'] == True):
          self.play_next()
          self.playhead = self.playhead + 1

          if (self.playhead / len(state['history'][0]) > trigger):  
            print("Generating more tokens ({} /{} > {})".format(self.playhead, len(state['history'][0]), trigger))
            seq = model.tick()[-128:]
            self.playhead = self.playhead - state['buffer_length'] + (len(seq) - len(state['history'][0]))
            state['history'][0] = seq

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

def play(note):
    state['scclient'].send_message('/play2', ['s', 'superpiano', 'note', note - NOTE_OFFSET])

def shutdown(unused_addr):
    stop_timer()
    #state['server'].server_close()
    state['server'].shutdown()
    unused_addr

def bind_dispatcher(dispatcher, model):
    state['model'] = model
    dispatcher.map("/start", engine_set, 'is_running', True)
    dispatcher.map("/pause", engine_set, 'is_running', False)
    dispatcher.map("/reset", lambda _: state['history'].clear())
    dispatcher.map("/end", shutdown)
    dispatcher.map("/quit", shutdown)
    dispatcher.map("/exit", shutdown)
    dispatcher.map("/debug", engine_print)
    dispatcher.map("/event", push_event)  # event2word

    dispatcher.map("/set", lambda addr, k, v: engine_set(addr, [k, v]))

    if (model):
        dispatcher.map("/sample", sample_model, model)
        dispatcher.map("/prep",   prepare_model, model)
    
    if (state['playback'] == True):
      dispatcher.map("/play", lambda _,note: play(note))

def load_folder(name):
  sys.path.append(os.path.join(sys.path[0], name))

def load_model():
  if state['model_name'] == 'music-transformer':
    load_folder('MusicTransformer-tensorflow2.0')
    from model import MusicTransformerDecoder
    import params as par
    return MusicTransformerDecoder(
        embedding_dim=256, vocab_size=par.vocab_size,
        num_layer=6,
        max_seq=state['max_seq'],
        dropout=0.1,
        debug=False)
  print("Unkown model: " + str(state['model_name'] + ". Aborting load..."))
  exit(-1)

# /TODO: Move to engine.py


# Main
if __name__ == "__main__":
    
    # Parse CLI Args
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_seq', default=256, help='최대 길이', type=int)

    parser.add_argument('--state', default="0", help='the initial state of the improv', type=str)
    parser.add_argument("--ip", default="127.0.0.1", help="The ip to listen on")
    parser.add_argument("--port", type=int, default=5005, help="The port to listen on")

    parser.add_argument("--model_name", type=str,  default="music-transformer", help="The model to use (music-transformer, remi)")
    parser.add_argument("--playback",   type=bool, default=True, help="Use supercollider for sound playback")
    parser.add_argument("--sc-ip",      type=str,  default="127.0.0.1", help="The supercollider server ip")
    parser.add_argument("--sc-port",    type=int,  default=57120, help="The supercollider server ip")

    args = parser.parse_args()

    state['model_name'] = args.model_name
    state['playback'] = args.playback
    state['scclient'] = udp_client.SimpleUDPClient(args.sc_ip, args.sc_port)
    state['max_seq'] = args.max_seq
    state['history'] = [[int(x) for x in str(args.state).split(',')]]


    # Prep Model
    model = load_model()
    prepare_model(None, [model])  # for real time use

    # Prep Server
    dispatcher = dispatcher.Dispatcher()
    bind_dispatcher(dispatcher, model)

    # Start Server
    state['server'] = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
    print("Serving {} on {}".format(state['model_name'], state['server'].server_address))
    start_timer()
    state['server'].serve_forever()
    stop_timer()
    model.close()