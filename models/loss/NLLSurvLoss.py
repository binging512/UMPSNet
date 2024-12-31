import torch

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
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
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + \
                                  torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))        # 
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def balanced_nll_loss(hazards, S, Y, c, alpha=0.4, beta=0.1, n_classes=4, eps=1e-7):
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
    hazards_padded = torch.cat([torch.zeros_like(c), hazards], 1)
    
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + \
                                  torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))        # 
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    
    unbalanced_loss = (1-alpha)* (censored_loss + uncensored_loss) + alpha* uncensored_loss
    
    
    balanced_uncensored_loss = -(1 - c) * (torch.log(
        torch.gather(S_padded, 1, Y).clamp(min=eps)*(torch.gather(1-hazards_padded, 1, Y).clamp(min=eps)**(n_classes-Y-1))) + \
                                  torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))        # 
    balanced_censored_loss = - c * torch.log(
        torch.gather(S_padded, 1, Y+1).clamp(min=eps)*(torch.gather(1-hazards_padded, 1, Y+1).clamp(min=eps)**(n_classes-Y-1)))
    
    balanced_loss = (1-alpha) * (balanced_censored_loss + balanced_uncensored_loss) + alpha * uncensored_loss
    
    loss = (1-beta)* unbalanced_loss+ beta*balanced_loss
    loss = loss.mean()
    return loss


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


class Balanced_NLLSurvLoss(object):
    def __init__(self, alpha=0.15, beta=0.5, n_classes= 4):
        self.alpha = alpha
        self.beta = beta
        self.n_classes = n_classes

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return balanced_nll_loss(hazards, S, Y, c, alpha=self.alpha, beta=self.beta, n_classes=self.n_classes)
        else:
            return balanced_nll_loss(hazards, S, Y, c, alpha=alpha, beta=self.beta, n_classes=self.n_classes)