import math
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from functools import partial
import torchmetrics.functional as tm_f
import torch.distributions as dist
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from torchmetrics import Metric
from torchmetrics.classification import MulticlassRecall, MulticlassPrecision
from icecream import ic

class CorrectAggregatedMetric(Metric):
    """This is needed to calculate some metrics b/c small batch sizes cause aggregation via a simple
        average to be off, as some classes might not be present in batch but will get penalized with a 0."""
    def __init__(self, class_idx: int, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.class_idx = torch.tensor(class_idx)
        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _update(self, numerator, denominator, preds, y) -> tuple:
        raise NotImplemented

    def update(self, logits: torch.Tensor, y: torch.Tensor):
        # update metric states
        preds = torch.argmax(logits, dim=-1)
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        assert preds.shape == y.shape, f"preds shape {preds.shape} != y shape {y.shape}"
        self.numerator, self.denominator = self._update(self.numerator, self.denominator, preds, y)

    def compute(self):
        # compute final result
        value = self.numerator.float() / self.denominator if self.denominator > 0 else torch.tensor(0.0)
        return value

    def reset(self):
        self.numerator = torch.tensor(0.0)
        self.denominator = torch.tensor(0.0)

class AccuracyPerClass(CorrectAggregatedMetric):
    """Calculate per class accuracy, i.e. P(y_hat = class_idx AND y = class_idx OR y_hat != class_idx AND y != class_idx)
    """
    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (y == class_idx)
        numerator += (preds[relevant_idxs] == class_idx).sum()
        denominator += relevant_idxs.sum()
        relevant_idxs = (y != class_idx)
        numerator += (preds[relevant_idxs] != class_idx).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator

class PrecisionPerClass(CorrectAggregatedMetric):
    """Calculate per class precision, i.e. P(y_hat = y | y_hat = class_idx)
    """
    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (preds == class_idx)
        numerator += (preds[relevant_idxs] == y[relevant_idxs]).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator


class RecallPerClass(CorrectAggregatedMetric):
    """Calculate per class recall, i.e. P(y_hat = y | y = class_idx)
    """
    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (y == class_idx)
        numerator += (preds[relevant_idxs] == y[relevant_idxs]).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator


def mcc(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return matthews_corrcoef(y.cpu().numpy(), y_hat.cpu().numpy())


def last_k_ppl(logits, y, seq_len=1024, k=None):
    '''
    Calculate perplexity for last k tokens in a sequence.

    logits: (batch_size * seq_len, vocab_size), note, already flattened
    y: (batch_size * seq_len), note, already flattened
    seq_len: int, length of each sequence in the batch
    k: if None, use all tokens in sequence
    
    returns: (batch_size,)  ppl for each sequence in the batch
    '''

    if k is None:
        k = 0  # use the entire sequence

    # need to reshape logits and y to be (batch_size, seq_len, vocab_size) and (batch_size, seq_len)
    # respectively
    # breakpoint()
    logits = logits.view(-1, seq_len, logits.shape[-1])
    y = y.view(-1, seq_len)

    # only use the last k values of seq dim in logits and y
    logits = logits[:, -k:, :]
    y = y[:, -k:]

    # reshape to flatten the batch and seq_len dimensions
    logits = logits.reshape(-1, logits.shape[-1])
    y = y.reshape(-1)
    # get avg and put on cpu
    return F.cross_entropy(logits, y, reduction='none').view(y.shape[0], -1).mean().exp().cpu()

def _student_t_map(mu, sigma, nu):
    sigma = F.softplus(sigma)
    nu = 2.0 + F.softplus(nu)
    return mu.squeeze(axis=-1), sigma.squeeze(axis=-1), nu.squeeze(axis=-1)

def student_t_loss(outs, y):
    mu, sigma, nu = outs[..., 0], outs[..., 1], outs[..., 2]
    mu, sigma, nu = _student_t_map(mu, sigma, nu)
    y = y.squeeze(axis=-1)

    nup1_half = (nu + 1.0) / 2.0
    part1 = 1.0 / nu * torch.square((y - mu) / sigma)
    Z = (
        torch.lgamma(nup1_half)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(math.pi * nu)
        - torch.log(sigma)
    )

    ll = Z - nup1_half * torch.log1p(part1)
    return -ll.mean()

def gaussian_ll_loss(outs, y):
    mu, sigma = outs[..., 0], outs[..., 1]
    y = y.squeeze(axis=-1)
    sigma = F.softplus(sigma)
    ll = -1.0 * (
        torch.log(sigma)
        + 0.5 * math.log(2 * math.pi)
        + 0.5 * torch.square((y - mu) / sigma)
    )
    return -ll.mean()

def binary_cross_entropy(logits, y):
    # BCE loss requires squeezing last dimension of logits so it has the same shape as y
    # requires y to be float, since it's overloaded to represent a probability
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())


def binary_accuracy(logits, y):
    return torch.eq(logits.squeeze(-1) >= 0, y).float().mean()

def padded_cross_entropy(logits, y, pad_mask, pad_value=-1):
    """Will ignore the pad value in label (eg, -1)
    
    logits: (batch_size, seq_len, vocab_size) #is actually (batch_size * seq_len, vocab_size)
    y: (batch_size, seq_len)
    pad_mask: (batch_size, seq_len)
    
    """
    
    #actually this is not even used, we only use the normal cross entropy
    
    # print('y_shape',y.shape)
    # print('y',y)
    # print('pad_mask',pad_mask[0:25])
    # need to apply pad mask to y
    y_pad = y + pad_mask * pad_value #this is just to make sure the pad value is ignored
    # import sys
    # sys.exit()
    logits = logits.view(-1, logits.shape[-1]) #all this does is flatten it to 1 dimension
    y_pad = y_pad.view(-1) #also 1 dimensions it, then calculates cross entropy loss
    return F.cross_entropy(logits, y_pad, ignore_index=pad_value)


def cross_entropy(logits, y, ignore_index=-100):
    # print('y before view', y.shape)
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    # print('y after view', y.shape) #is actually exactly the same
    # print('y', y)
    # import sys
    # sys.exit()
    
    #let's visualize the logits a bit better
    # print('example 15', logits[15])
    
    return F.cross_entropy(logits, y, ignore_index=ignore_index)


def soft_cross_entropy(logits, y, label_smoothing=0.0):
    logits = logits.view(-1, logits.shape[-1])
    # target is now 2d (no target flattening)
    return F.cross_entropy(logits, y, label_smoothing=label_smoothing)


def accuracy(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    preds = torch.argmax(logits, dim=-1)
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.eq(preds, y).float().mean()


def accuracy_ignore_index(logits, y, ignore_index=-100):
    num_classes = logits.shape[-1]
    preds = torch.argmax(logits, dim=-1)
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    accuracy = tm_f.classification.accuracy(preds, y, 'multiclass', num_classes=num_classes, ignore_index=ignore_index, average='micro')
    return accuracy


def accuracy_at_k(logits, y, k=1):
    logits = logits.view(-1, logits.shape[-1])
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.topk(logits, k, dim=-1)[1].eq(y.unsqueeze(-1)).any(dim=-1).float().mean()


def f1_binary(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="binary")


def f1_macro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="macro")


def f1_micro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="micro")


def roc_auc_macro(logits, y):
    logits = logits.view(
        -1, logits.shape[-1]
    ).detach()  # KS: had to add detach to eval while training
    y = y.view(-1)
    return roc_auc_score(
        y.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()[:, 1], average="macro"
    )


def roc_auc_micro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return roc_auc_score(
        y.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()[:, 1], average="micro"
    )


def mse(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    # y = y.squeeze(-1)
    # ic(outs)
    # ic(len(y))
    # ic(y)
    # ic(outs.shape)
    # # ic(y.shape)
    # ic(y[0].shape)
    # ic(y[1].shape)
    # ic(len_batch)
    # import sys
    # sys.exit()
    #check if outs is a tuple
    if isinstance(outs, tuple): #for profile prediction, it's the second output
        outs = outs[1]
        y = y[1]
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.mse_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        # TODO document the use case of this
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.mse_loss(outs_masked, y_masked)

def forecast_rmse(outs, y, len_batch=None):
    # TODO: generalize, currently for Monash dataset
    return torch.sqrt(F.mse_loss(outs, y, reduction='none').mean(1)).mean()

def mae(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    outs = outs.squeeze(-1) #inputs are actually shape (batch_size, 1)
    y = y.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.l1_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.l1_loss(outs_masked, y_masked)

def binary_cross_entropy_with_logits(logits, y, ignore_index = -100, pos_weight=.04):
    # No need to reshape if your logits and y are already in the desired shape (batch_size, num_nodes)
    pos_weight_value = pos_weight #no need to copy it, it's immutable
    num_classes = logits.shape[1]
    pos_weight = torch.full((num_classes,), pos_weight_value, dtype=torch.float).to(logits.device)
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1, logits.shape[-1])
    #make sure y is type float
    y = y.float()
    # ic(y) #1s and 0s
    # ic(logits) # outputs of model
    # ic(logits.shape) # batch x 161 for multitasking
    # ic(y.shape) # batch x 161 for multitasking
    # ic(pos_weight.shape) #[161] for multitasking
    # ic(pos_weight) # all just .04
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)/(2*pos_weight_value) #the 2 is because there's 2 elements in the sum, keeps it similar scales

def custom_mse_ce(outs, y, len_batch=None, ignore_index=-100, mask = True, weight = .5, pos_weight=.04):
    '''This loss function will be used for the custom model for the DNase dataset, the goal is to minimize the MSE and the CE at the same time'''
    #first has to split the outs into the two different outputs
    #it's always going to be exactly half
    #len batch is indeed none, just ignore that
    # if outs.shape[1] == 2:
    #     class_out = outs[:,0]
    #     reg_out = outs[:,1]
    #     class_y = y[0]
    #     reg_y = y[1]
    # else:        
    class_out = outs[:,0:outs.shape[1]//2] #this should work just fine for either 2 or 322
    reg_out = outs[:,outs.shape[1]//2:]
    class_y = y[0]
    reg_y = y[1] #should be 64x1 or 64 x 161 if multitasking

    # ic(class_out.shape)
    # ic(reg_out.shape)
    # ic(class_y.shape)
    # ic(reg_y.shape)
    # import sys
    # sys.exit()

    #now that we have the two different outputs, we can calculate the two different losses
    #note that we also need to know what len_batch is, so see where this may be called?
    #or we literally just use their functions
    ce_loss = binary_cross_entropy_with_logits(class_out, class_y, pos_weight=pos_weight)
    #now we mask the regression so that it ignores the values where it 
    if mask:
        mask = class_y == 1 #finds where it's 1, and keeps thos
        #now apply the mask and set those rows to 0
        reg_out = reg_out * mask
        reg_y = reg_y * mask
    mse_loss = mse(reg_out, reg_y, len_batch)
    #the other factor is that if ce loss is too high, then it will dominate the loss, so we need to scale it
    #now we just add them together
    # ic(ce_loss)
    # ic(mse_loss)
    # ic(weight*mse_loss + (1-weight)*ce_loss)
    # ic(ignore_index)
    # ic(len_batch)
    # import sys
    # sys.exit()
    return weight*mse_loss + (1-weight)*ce_loss #note we can weight them if we want to

def custom_mse(outs, y, len_batch=None, ignore_index=-100, mask = True, weight = .5):
    class_out = outs[:,0:outs.shape[1]//2] #this should work just fine for either 2 or 322
    reg_out = outs[:,outs.shape[1]//2:]
    class_y = y[0]
    reg_y = y[1] #should be 64x1 or 64 x 161 if multitasking
    if mask:
        mask = class_y == 1 #finds where it's 1, and keeps thos
        #now apply the mask and set those rows to 0
        reg_out = reg_out * mask
        reg_y = reg_y * mask
    mse_loss = mse(reg_out, reg_y, len_batch)
    return mse_loss*weight

def custom_ce(outs, y, len_batch=None, ignore_index=-100, mask = True, weight = .5, pos_weight = .04):
    class_out = outs[:,0:outs.shape[1]//2] #this should work just fine for either 2 or 322
    reg_out = outs[:,outs.shape[1]//2:]
    class_y = y[0]
    reg_y = y[1] #should be 64x1 or 64 x 161 if multitasking
    ce_loss = binary_cross_entropy_with_logits(class_out, class_y, pos_weight=pos_weight)
    return ce_loss*(1-weight)

def cbpnet_multinomial_nll(logits,true_counts, len_batch=None, ignore_index=-100, mask = True):

    """A loss function based on the multinomial negative log-likelihood.
    modified by me to include things like doing thE log softmax and deal with more complex tracking

    This loss function takes in a tensor of normalized log probabilities such
    that the sum of each row is equal to 1 (e.g. from a log softmax) and
    an equal sized tensor of true counts and returns the probability of
    observing the true counts given the predicted probabilities under a
    multinomial distribution. Can accept tensors with 2 or more dimensions
    and averages over all except for the last axis, which is the number
    of categories.

    Adapted from Alex Tseng.

    Parameters
    ----------
    logps: torch.tensor, shape=(n, ..., L)
        A tensor with `n` examples and `L` possible categories. 

    true_counts: torch.tensor, shape=(n, ..., L)
        A tensor with `n` examples and `L` possible categories.

    Returns
    -------
    loss: float
        The multinomial log likelihood loss of the true counts given the
        predicted probabilities, averaged over all examples and all other
        dimensions.
    """
    if isinstance(logits, tuple): #for wandb tracking, it inputs logits here, since then it's a tuple
        logits = logits[0]
        true_counts = true_counts[0]
    #we also need to make sure that we have the right shape
    logits = logits.squeeze(); true_counts = true_counts.squeeze()
    # if logits.shape[1] > true_counts.shape[1]: #added it in the tasks instead
    #     #then we cut off from both ends until we get the same size
    #     diff = logits.shape[1] - true_counts.shape[1]
    #     start = diff // 2
    #     end = start + true_counts.shape[1]
    #     logits = logits[:, start:end]
        # logits = logits[:,diff//2:-diff//2]
    logps = torch.log_softmax(logits, dim=-1)
    log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1)
    log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1)
    log_prod_exp = torch.sum(true_counts * logps, dim=-1)
    return (-log_fact_sum + log_prod_fact - log_prod_exp).mean()
    

def custom_profile_loss(outs, y, len_batch=None, ignore_index=-100, mask = True, count_weight = 3.6):
    '''
    based on chrombpnet implementation of this loss
    '''
    profile = outs[0]
    counts = outs[1]
    label_profile = y[0]
    label_counts = y[1]
    mse_loss = mse(counts, label_counts, len_batch)
    multinomial_loss = cbpnet_multinomial_nll(profile, label_profile)
    return count_weight*mse_loss + multinomial_loss

# Metrics that can depend on the loss
def loss(x, y, loss_fn):
    """ This metric may be useful because the training loss may add extra regularization (e.g. weight decay implemented as L2 penalty), while adding this as a metric skips the additional losses """
    return loss_fn(x, y)


def bpb(x, y, loss_fn):
    """ bits per byte (image density estimation, speech generation, char LM) """
    return loss_fn(x, y) / math.log(2)


def ppl(x, y, loss_fn):
    return torch.exp(loss_fn(x, y))


# should have a better way to do this
output_metric_fns = {
    "binary_cross_entropy": binary_cross_entropy,
    "cross_entropy": cross_entropy,
    "padded_cross_entropy": padded_cross_entropy,
    "binary_accuracy": binary_accuracy,
    "precision": MulticlassPrecision,
    "precision_per_class": PrecisionPerClass,
    "recall": MulticlassRecall,
    "recall_per_class": RecallPerClass,
    "accuracy": accuracy,
    "accuracy_per_class": AccuracyPerClass,
    "accuracy_ignore_index": accuracy_ignore_index,
    'accuracy@3': partial(accuracy_at_k, k=3),
    'accuracy@5': partial(accuracy_at_k, k=5),
    'accuracy@10': partial(accuracy_at_k, k=10),
    "eval_loss": loss,
    "mcc": mcc,
    "mse": mse,
    "mae": mae,
    "forecast_rmse": forecast_rmse,
    "f1_binary": f1_binary,
    "f1_macro": f1_macro,
    "f1_micro": f1_micro,
    "roc_auc_macro": roc_auc_macro,
    "roc_auc_micro": roc_auc_micro,
    "soft_cross_entropy": soft_cross_entropy,  # only for pytorch 1.10+
    "student_t": student_t_loss,
    "gaussian_ll": gaussian_ll_loss,
    "custom_mse_ce": custom_mse_ce,
    "custom_mse": custom_mse,
    "custom_ce": custom_ce,
    'custom_profile_loss': custom_profile_loss,
    'cbpnet_multinomial_nll': cbpnet_multinomial_nll,
}

loss_metric_fns = {
    "loss": loss,
    "bpb": bpb,
    "ppl": ppl,
}
metric_fns = {**output_metric_fns, **loss_metric_fns}  # TODO py3.9

