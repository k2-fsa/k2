import torch
from torch.distributions.categorical import Categorical
from typing import Tuple

def importance_sampling(
    sampling_scores: torch.Tensor,
    path_length: int,
    num_paths: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
      sampling_scores:
        The output of predictor head, a tensor of shape (B, S, T, V) containing
        the probabilities of emitting symbols at each (t, s) for each sequence.
      path_length:
        How many symbols we will sample for each path.
      num_paths:
        How many paths we will sample for each sequence.

    Returns:
      Three tensors will be returned.
      - sampled_indexs:
        A tensor of shape (B, num_paths, path_length), containing the sampled symbol ids.
      - sampled_scores:
        A tensor of shape (B, num_paths, path_length), containing the sampling probabilities
        of the corresponding symbols.
      - sampled_t_indexs:
        A tensor of shape (B, num_paths, path_length), containing the frame ids, which means
        at what frame this symple be sampled.
    """
    (B, S, T, V) = sampling_scores.shape
    # we sample paths from frame 0
    t_index = torch.zeros(
        (B, num_paths),
        dtype=torch.int64,
        device=sampling_scores.device
    )
    # we sample paths from the first symbols (i.e. from null left_context)
    s_index = torch.zeros(
        (B, num_paths),
        dtype=torch.int64,
        device=sampling_scores.device
    )

    sampled_indexs = []
    sampled_scores = []
    sampled_t_indexs = []

    for i in range(path_length):
        # select context symbols for paths
        # sub_scores : (B, num_paths, T, V)
        sub_scores = torch.gather(
            sampling_scores, dim=1,
            index=s_index.reshape(B, num_paths, 1, 1).expand(B, num_paths, T, V))

        # select frames for paths
        # sub_scores : (B, num_paths, 1, V)
        sub_scores = torch.gather(
            sub_scores, dim=2,
            index=t_index.reshape(B, num_paths, 1, 1).expand(B, num_paths, 1, V))

        # sub_scores : (B, num_paths, V)
        sub_scores = sub_scores.squeeze(2)
        # sampler: https://pytorch.org/docs/stable/distributions.html#categorical
        sampler = Categorical(probs=sub_scores)

        # sample one symbol for each path
        # index : (B, num_paths)
        index = sampler.sample()
        sampled_indexs.append(index)

        # gather sampling probabilities for corresponding indexs
        # score : (B, num_paths, 1)
        score = torch.gather(sub_scores, dim=2, index=index.unsqueeze(2))
        sampled_scores.append(score.squeeze(2))

        sampled_t_indexs.append(t_index)

        # update (t, s) for each path (for regular RNN-T)
        # index == 0 means the sampled symbol is blank
        t_mask = index == 0
        t_index = torch.where(t_mask, t_index + 1, t_index)
        s_index = torch.where(t_mask, s_index + 1, s_index)

    # indexs : (B, num_paths, path_lengths)
    indexs = torch.stack(sampled_indexs, dim=0).permute(1,2,0)
    # scores : (B, num_paths, path_lengths)
    scores = torch.stack(sampled_scores, dim=0).permute(1,2,0)
    # t_indexs : (B, num_paths, path_lengths)
    t_indexs = torch.stack(sampled_t_indexs, dim=0).permute(1,2,0)

    return indexs, scores, t_indexs


if __name__ == "__main__":
    B, S, T, V = 2, 10, 20, 10
    path_length = 8
    num_path = 3
    logits = torch.randn((B, S, T, V))
    log_prob = torch.softmax(logits, -1)
    indexs, scores, t_indexs = importance_sampling(log_prob, path_length, num_path)
    print (indexs)
    print (scores)
    print (t_indexs)
