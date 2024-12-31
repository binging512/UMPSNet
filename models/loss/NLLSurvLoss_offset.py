import torch

def nll_loss(hazards, S, Y, c, offset,alpha=0.4, eps=1e-7,num_classes=8):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    survival_bin = Y - offset  # 从标签中剥离出生存时间区间

    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + \
                                  torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))        # 
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss



class NLLSurvLoss_offset(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, offset,alpha=None,num_classes=8):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, offset=offset, alpha=self.alpha, num_classes=num_classes)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)