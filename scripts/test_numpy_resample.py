
import numpy as np
import wave
import time

class LinearResampler2x:
    def __init__(self):
        self.last_sample = None
        
    def resample(self, chunk: np.ndarray) -> np.ndarray:
        # chunk: (N,) int16
        if len(chunk) == 0: return np.array([], dtype=np.int16)
        
        # Convert to float for calculation (optional, but safer for overflow)
        # Actually, let's keep it simple.
        
        # We want to insert points.
        # Output len = 2 * N
        # If we have last_sample:
        #   First output is mid(last, chunk[0]), NO wait.
        #   Standard Linear Interp 2x:
        #   Original: A, B, C
        #   Target: A, (A+B)/2, B, (B+C)/2, C...
        
        # Better alignment for continuous streams:
        # We need to output 2 samples for every 1 input sample to maintain sync?
        # Or is it (N-1)*2 + 1?
        # For a stream, we effectively want to treat it as continuous.
        # Rate L -> 2L.
        # Time 0: A -> Time 0: A
        # Time 0.5: (A+B)/2
        # Time 1: B -> Time 1: B
        
        out = np.zeros(len(chunk) * 2, dtype=np.int16)
        
        # Even indices (0, 2, 4...) = Original samples
        out[0::2] = chunk
        
        # Odd indices (1, 3, 5...) = Midpoints
        # out[1] = (chunk[0] + chunk[1]) / 2
        
        # We need chunk shifted by 1.
        # chunk[:-1] and chunk[1:] gives pairs (0,1), (1,2)...
        
        # Calculation:
        # We need to handle the boundary: "Previous Chunk's Last Sample" -> "Current Chunk's First Sample"
        
        chunk_float = chunk.astype(np.float32)
        
        interp = np.zeros(len(chunk), dtype=np.float32)
        
        # Midpoints internal to the chunk
        # interp[i] corresponds to out[2*i - 1]? No.
        # out[1] is between chunk[0] and chunk[1].
        # out[2i+1] is between chunk[i] and chunk[i+1].
        
        # General case (internal):
        # mids = (chunk[:-1] + chunk[1:]) / 2
        
        # Handle first element:
        # out[1] needs chunk[0] and chunk[1].
        # out[-1] needs chunk[-1] and NEXT chunk[0]? 
        # No, a 2x upsampler usually has latency or assumes casual.
        
        # Let's say:
        # Output[2i] = Input[i]
        # Output[2i+1] = (Input[i] + Input[i+1]) / 2
        
        # This requires lookahead of 1 sample (Input[i+1]).
        # Or delay of 1 output sample.
        
        # If we don't want delay, we can extrapolate, but that's bad.
        # Delay 1 input sample = Delay 2 output samples.
        
        # State:
        # We must hold `state` = the last sample of previous chunk, to interpolate between state and chunk[0].
        
        # Wait, if out[2i] = In[i], then out[0] = In[0].
        # out[-1] (last element) would be (In[-1] + Next_In[0])/2.
        # We don't have Next_In[0].
        # So we cannot output the very last interpolated sample yet?
        
        # Alternative: 
        # Output[2i] = In[i]
        # Output[2i-1] = (In[i-1] + In[i]) / 2
        
        # When chunk arrives:
        # Loop i from 0 to N-1
        # out[2i] = chunk[i]
        # out[2i-1]... wait.
        
        # Let's define the odd samples:
        # out[1] = (chunk[0] + chunk[1]) / 2
        # out[3] = (chunk[1] + chunk[2]) / 2
        # ...
        # out[2*i + 1] = (chunk[i] + chunk[i+1]) / 2
        
        # For the last sample i=N-1:
        # out[2*(N-1)] = chunk[N-1]
        # out[2*(N-1) + 1] = (chunk[N-1] + NEXT_CHUNK[0]) / 2  <-- Missing.
        
        # So we hold back chunk[N-1] to be the "state" for the next call?
        # Yes.
        
        # Algorithm:
        # buffer = [self.last_sample] + list(chunk)
        # But wait, if self.last_sample is None (first chunk), what?
        # Just assume 0? or duplicate first sample?
        
        if self.last_sample is None:
            # First chunk
            # Replicate first sample to "fill" the backward gap? 
            # Or just output 2*N - 1 samples? 
            # Ideally expected output for T input is 2T output.
            
            # Let's pretend previous was 0.
            prev_last = 0 
            # Actually, to align with "no pops", maybe assume 0.
        else:
            prev_last = self.last_sample
            
        # We need to output N pairs. 
        # Pair i:
        #   Sample A: (prev_last + chunk[0]) / 2   <-- Wait, where does this go?
        #   Sample B: chunk[0]
        
        # Let's map usage.
        # t= -1: prev_last
        # t= 0: chunk[0]
        # t= 1: chunk[1]
        
        # We want output at t=0, t=0.5, t=1.0...
        # t=0: chunk[0]
        # t=0.5: (chunk[0]+chunk[1])/2
        # t=1: chunk[1]
        
        # So we need to carry over the *last* sample of the chunk to the next call, 
        # and NOT output the interpolated value for the end of the chunk yet.
        
        # Input: N samples.
        # Output: 2*N samples.
        
        # We can implement a 1-sample delay.
        # We process (prev_last, chunk[:-1]).
        # Save chunk[-1] as new prev_last.
        
        full_seq = np.concatenate(([prev_last], chunk)) # Length N+1
        
        # We produce output for indices 0 to N-1 of original stream?
        # No, we want to clear the buffer.
        
        # Simple Logic:
        # For each sample x in chunk:
        #   Output 1: (prev + x) / 2
        #   Output 2: x
        #   prev = x
        
        # Let's trace:
        # Seq: A, B, C
        # 1. (0+A)/2, A
        # 2. (A+B)/2, B
        # 3. (B+C)/2, C
        # Result: 0.5A, A, 0.5(A+B), B, 0.5(B+C), C
        # Is this correct 2x upsampling?
        # Original: A . B . C
        # Interp:   A ab B bc C
        # Mine:    .5A A ab B bc C
        # It introduces a half-sample shift and a startup artifact (.5A).
        # But it preserves continuity!
        # And it yields exactly 2*N samples.
        
        # Let's stick with this. It's a "Linear Interpolator with 0.5 sample delay".
        # 24k -> 48k is high enough rate that 0.5 sample shift is audible? No. 1/48000s = 20us.
        
        # Vectorized:
        # prev_arr = full_seq[:-1]
        # curr_arr = full_seq[1:]
        
        # out_interp = (prev_arr + curr_arr) / 2
        # out_raw    = curr_arr
        
        # We want to interleave them.
        # out[0::2] = out_interp
        # out[1::2] = out_raw
        
        full_seq = np.concatenate(([prev_last], chunk)).astype(np.float32)
        
        interp = (full_seq[:-1] + full_seq[1:]) * 0.5
        raw = full_seq[1:]
        
        out = np.empty(len(chunk) * 2, dtype=np.int16)
        out[0::2] = interp.astype(np.int16)
        out[1::2] = raw.astype(np.int16)
        
        self.last_sample = chunk[-1]
        return out

def test_numpy_resampler():
    # Load 1 second of our ref file or generated sine wave
    rate = 24000
    t = np.linspace(0, 1, rate, endpoint=False)
    freq = 440
    # Generate continuous sine
    audio = (np.sin(2 * np.pi * freq * t) * 32000).astype(np.int16)
    
    # Split into chunks to simulate streaming
    chunk_size = 1024
    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
    
    resampler = LinearResampler2x()
    full_output = []
    
    start = time.time()
    for c in chunks:
        out = resampler.resample(c)
        full_output.append(out)
    end = time.time()
    
    full_output = np.concatenate(full_output)
    
    print(f"Processed {len(audio)} samples in {end-start:.4f}s")
    print(f"Output size: {len(full_output)} (Expected: {len(audio)*2})")
    
    # Save for inspection
    with wave.open("numpy_resample_test.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(48000)
        f.writeframes(full_output.tobytes())
        
    print("Saved numpy_resample_test.wav")

if __name__ == "__main__":
    test_numpy_resampler()
