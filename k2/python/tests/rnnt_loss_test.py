import random
import torch
import k2


def test_rnnt_logprobs_basic():
    print("Running test_rnnt_logprobs_basic()")

    B = 1
    S = 3
    T = 4
    C = 3

    # lm: [B][S+1][C]
    lm = torch.tensor([[[0, 0, 1], [0, 1, 1], [1, 0, 1], [2, 2, 0]]],
                      dtype=torch.float)
    # am: [B][T][C]
    am = torch.tensor([[[0, 1, 2], [0, 0, 0], [0, 2, 4], [0, 3, 3]]],
                      dtype=torch.float)

    #    lm[:] = 0.0
    #    am[:] = 0.0

    termination_symbol = 2
    symbols = torch.tensor([[0, 1, 0]], dtype=torch.long)

    px, py = k2.get_rnnt_logprobs(lm, am, symbols, termination_symbol)

    assert px.shape == (B, S, T + 1)
    assert py.shape == (B, S + 1, T)
    assert symbols.shape == (B, S)
    print("px = ", px)
    print("py = ", py)
    m, grad = k2.mutual_information_recursion(px, py)
    print("m = ", m)

    t_am = am.unsqueeze(2)
    t_lm = lm.unsqueeze(1)
    t_prob = t_am + t_lm
    m1, grad = k2.rnnt_loss(t_prob, symbols, termination_symbol, None)
    print("m1 = ", m1)
    assert torch.allclose(m, m1)

    # should be invariant to adding a constant for any frame.
    lm += torch.randn(B, S + 1, 1)
    am += torch.randn(B, T, 1)

    m2, grad = k2.rnnt_loss_simple(lm, am, symbols, termination_symbol, None)
    print("m2 = ", m2)
    assert torch.allclose(m, m2)

    t_am = am.unsqueeze(2)
    t_lm = lm.unsqueeze(1)
    t_prob = t_am + t_lm
    m3, grad = k2.rnnt_loss(t_prob, symbols, termination_symbol, None)
    print("m3 = ", m3)
    assert torch.allclose(m, m3)

    device = torch.device('cuda')
    m4, grad = k2.rnnt_loss_simple(lm.to(device), am.to(device),
                                   symbols.to(device), termination_symbol,
                                   None)
    print("m4 = ", m4)

    m5, grad = k2.rnnt_loss_aux(lm.to(device),
                                am.to(device),
                                symbols.to(device),
                                termination_symbol,
                                lm_only_scale=0.0,
                                am_only_scale=0.0,
                                boundary=None)
    print("m5 = ", m5)

    assert torch.allclose(m, m3.to('cpu'))

    assert torch.allclose(m, m4.to('cpu'))

    assert torch.allclose(m, m5.to('cpu'))


def test_rnnt_logprobs_aux():

    print("Running test_rnnt_logprobs_aux()")

    B = 1
    S = 3
    T = 4
    C = 3

    # lm: [B][S+1][C]
    lm = torch.tensor([[[0, 0, 1], [0, 1, 1], [1, 0, 1], [2, 2, 0]]],
                      dtype=torch.float)
    # am: [B][T][C]
    am = torch.tensor([[[0, 1, 2], [0, 0, 0], [0, 2, 4], [0, 3, 3]]],
                      dtype=torch.float)

    termination_symbol = 2
    symbols = torch.tensor([[0, 1, 0]], dtype=torch.long)

    device = torch.device('cuda')
    m1, grad = k2.rnnt_loss_aux(lm.to(device),
                                am.to(device),
                                symbols.to(device),
                                termination_symbol,
                                lm_only_scale=0.0,
                                am_only_scale=0.333,
                                boundary=None)
    print("m1 = ", m1)

    # should be invariant to adding a constant for any frame.
    lm += torch.randn(B, S + 1, 1)
    am += torch.randn(B, T, 1)

    m2, grad = k2.rnnt_loss_aux(lm.to(device),
                                am.to(device),
                                symbols.to(device),
                                termination_symbol,
                                lm_only_scale=0.0,
                                am_only_scale=0.333,
                                boundary=None)
    print("m2 = ", m2)

    assert torch.allclose(m1, m2)


if __name__ == "__main__":
    test_rnnt_logprobs_aux()
    test_rnnt_logprobs_basic()
