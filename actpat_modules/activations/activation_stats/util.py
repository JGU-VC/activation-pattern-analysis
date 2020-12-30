import torch
import numpy as np

class DistributeGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, counts, bins_to_act_indices):
        ctx.save_for_backward(bins_to_act_indices)
        ctx.num_dims  = input.shape[1]
        ctx.input_shape = input.shape
        return counts.type(torch.float)

    @staticmethod
    def backward(ctx, grad_counts):
        bins_to_act_indices, = ctx.saved_tensors

        # no gradients if no counts given
        if len(grad_counts) == 0:
            return None, None, None, None

        # match the input with the corresponding gradients of counts
        grad_counts_per_input = grad_counts[bins_to_act_indices].expand(ctx.input_shape)

        return grad_counts_per_input, None, None

# bin -> numbers
def bin2numbers(input):
    numbers = []
    for i in range((input.size(1)-1)//64 + 1):
        block = input[:,i*64:(i+1)*64]
        base = 2**torch.arange(block.size(1), device=input.device)

        # ensure that base is at correct dimension
        base = base.unsqueeze(0)
        for i in range(len(input.shape)-2):
            base = base.unsqueeze(i+2)

        # pytorch does not support uint64_t, but int64_t
        # thus, we interpret last bit as sign
        if block.size(1) == 64:
            base[:,-1] = -1

        num = (block*base).sum(1)
        numbers.append(num)
    return torch.stack(numbers, dim=1)

def hash(act, labels=None):
    FIRSTH = 37 # prime
    A = 54059 # a prime
    B = 76963 # another prime
    # C = 86969 # yet another prime

    device = act.device

    numbers = bin2numbers(act)
    if labels is not None:
        numbers = torch.cat([numbers,labels],1)
    numbers = numbers * B
    result = FIRSTH;
    d = numbers.shape[1];
    for i in range(d):
        number = numbers[:,i]
        result = number ^ (result * A);
    return result;


def get_hist_torch(act, labels=None, get_bins=False, N=30000, device="same", dtype=torch.int32):
    counts = torch.zeros(N, dtype=dtype, device=act.device if device == "same" else device)
    if device != "same":
        act = act.to(device)
    hashes = hash(act, labels=labels)
    bins_to_act_indices = hashes % N
    # if device != "same":
    #     bins_to_act_indices = bins_to_act_indices.to(device)
    n = np.prod(bins_to_act_indices.shape)
    counts.index_add_(0, bins_to_act_indices.view(-1), torch.ones(1,device=bins_to_act_indices.device,dtype=dtype).expand(n))
    return bins_to_act_indices.unsqueeze(1), counts
