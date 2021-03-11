from model import MusicTransformerDecoder
from midi_processor.processor import Event
import params as par

class OrnetteModule(MusicTransformerDecoder):
    def __init__(self, state={}, checkpoint='unconditional_model_16'):
      super().__init__(embedding_dim=256,
        vocab_size=par.vocab_size,
        num_layer=6,
        max_seq=state['max_seq'],
        dropout=0.1)
      self.realtime_ready = False
      self.server_state = state


    def realtime_setup(self,state):
      """Initializes internal model state to be used in real-time

        Parameters:
        state (dict): Ornette Server state, passed via server

        Returns: Nothing
      """
      if (self.realtime_ready == True):
          return
          
      # self.server_state = state
      self.realtime_ready = True
    
    def tick(self):
      # prior = self.server_state['history'][0][-16:]
      prior = self.server_state['history'][0]
      buffer_length = self.server_state['buffer_length']
      seq = self.generate(prior, length=buffer_length)
      
      # Update clock (next request)
      # total_time = sum([e.value for e in map(Event.from_int, seq) if e.type == 'time_shift'])
      # self.server_state['tick_interval'] = total_time / 1000

      return seq

    def decode(self, token):
      e = Event.from_int(token)
      return (e.type,e.value)

    def close():
      pass
