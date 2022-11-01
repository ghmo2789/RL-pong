from torch._C._autograd import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile

import train

if __name__ == "__main__":
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            train.run()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
