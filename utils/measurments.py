import numpy as np
import torch
from torch import nn


def measure_eval_time(model: nn.Module, dummy_input: torch.Tensor) -> tuple[float, float]:
    """
    Measure the time of evaluating model or module.
    Args:
        model: Any model/classifier
        dummy_input: tensor of size accepted by given model

    Returns:
        tuple of mean time, measured in milliseconds and its standard deviation
    """
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    dummy_input = dummy_input.to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    with torch.no_grad():
        # GPU-WARM-UP
        for _ in range(100):
            _ = model(dummy_input)
        # MEASURE PERFORMANCE
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings).item()
    return mean_syn, std_syn

