#!/usr/bin/env python3
#
# Copyright      2022  Xiaomi Corporation      (authors: Liyong Guo)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To run this single test, use
#
#  ctest --verbose -R mwer_test_py

import unittest

import k2
import torch


class TestMWERLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def _common_test_part(self, reduction, logp, logp_lattice):
        # Note:
        #  log(0.1) == -2.3026
        #  log(0.2) == -1.6094
        #  log(0.3) == -1.2040
        #  log(0.4) == -0.9163
        #  log(0.5) == -0.6931
        for device in self.devices:
            logp = logp.to(device)
            logp_lattice = logp_lattice.to(device)
            s = """
                0 1 1 10 -2.3026
                0 1 5 10 -0.6931
                0 1 2 20 -1.6094
                1 2 3 30 -1.2040
                1 2 4 40 -0.9163
                2 3 -1 -1 -0.6931
                3
            """
            lattice = k2.Fsa.from_str(s, acceptor=False)
            lattice = k2.Fsa.from_fsas([lattice, lattice])
            lattice.scores = logp_lattice

            refs_texts = [[10], [50]]
            loss = k2.mwer_loss(lattice, refs_texts,
                                nbest_scale=1.0, num_paths=200,
                                reduction=reduction)

            # each lattice has 4 distinct paths
            # that have different word sequences:
            # for lattice[0]:
            # path       scores      path_p    normalized    hyps    refs  wers
            # 10->30  [0.1 0.3 0.5]  0.015   0.015 / 0.105  [10, 30] [10]   1
            # 10->40  [0.1 0.4 0.5]  0.020   0.020 / 0.105  [10, 40] [10]   1
            # 20->30  [0.2 0.3 0.5]  0.030   0.030 / 0.105  [20, 30] [10]   2
            # 20->40  [0.2 0.4 0.5]  0.040   0.040 / 0.105  [20, 40] [10]   2
            #
            # for lattice[1]:
            # path       scores      path_p    normalized    hyps    refs  wers
            # 10->30  [0.1 0.3 0.5]  0.015   0.015 / 0.105  [10, 30] [50]   2
            # 10->40  [0.1 0.4 0.5]  0.020   0.020 / 0.105  [10, 40] [50]   2
            # 20->30  [0.2 0.3 0.5]  0.030   0.030 / 0.105  [20, 30] [50]   2
            # 20->40  [0.2 0.4 0.5]  0.040   0.040 / 0.105  [20, 40] [50]   2

            # path_i_j is short path i from lattice j.
            path_0_0_logp = logp[0] + logp[3] + logp[5]
            path_1_0_logp = logp[0] + logp[4] + logp[5]
            path_2_0_logp = logp[2] + logp[3] + logp[5]
            path_3_0_logp = logp[2] + logp[4] + logp[5]
            path_0_1_logp = logp[6] + logp[9] + logp[11]
            path_1_1_logp = logp[6] + logp[10] + logp[11]
            path_2_1_logp = logp[8] + logp[9] + logp[11]
            path_3_1_logp = logp[8] + logp[10] + logp[11]

            den_0_logp = torch.logsumexp(torch.stack([path_0_0_logp,
                                                      path_1_0_logp,
                                                      path_2_0_logp,
                                                      path_3_0_logp,
                                                      ]), dim=0)
            den_1_logp = torch.logsumexp(torch.stack([path_0_1_logp,
                                                      path_1_1_logp,
                                                      path_2_1_logp,
                                                      path_3_1_logp,
                                                      ]), dim=0)

            prob_0_0 = (path_0_0_logp - den_0_logp).exp()
            prob_1_0 = (path_1_0_logp - den_0_logp).exp()
            prob_2_0 = (path_2_0_logp - den_0_logp).exp()
            prob_3_0 = (path_3_0_logp - den_0_logp).exp()
            prob_0_1 = (path_0_1_logp - den_1_logp).exp()
            prob_1_1 = (path_1_1_logp - den_1_logp).exp()
            prob_2_1 = (path_2_1_logp - den_1_logp).exp()
            prob_3_1 = (path_3_1_logp - den_1_logp).exp()

            prob_normalized = torch.stack([prob_0_0, prob_1_0,
                                           prob_2_0, prob_3_0,
                                           prob_0_1, prob_1_1,
                                           prob_2_1, prob_3_1])
            wers = torch.tensor([1, 1, 2, 2, 2, 2, 2, 2]).to(device)
            prob_normalized = prob_normalized * wers
            return loss, prob_normalized

    def test(self):
        for reduction in ['sum', 'mean', 'none']:
            # used to verify gradient.
            prob = torch.tensor([0.1, 0.5, 0.2,
                                 0.3, 0.4, 0.5]).repeat(2)
            prob.requires_grad_()
            logp = prob.log()

            # assigned to lattice.scores
            prob_lattice = torch.tensor([0.1, 0.5, 0.2,
                                         0.3, 0.4, 0.5]).repeat(2)
            prob_lattice.requires_grad_()
            logp_lattice = prob_lattice.log()

            # test reduction == 'sum'
            loss, prob_normalized = self._common_test_part(reduction,
                                                           logp,
                                                           logp_lattice)
            loss_sum_expected = prob_normalized.sum()
            loss_mean_expected = prob_normalized.mean()
            if reduction == 'sum':
                assert torch.isclose(loss, loss_sum_expected.to(loss.dtype))
                loss.backward()
                loss_sum_expected.backward()
                assert torch.allclose(prob.grad, prob_lattice.grad, atol=1e-5)
            elif reduction == 'mean':
                assert torch.isclose(loss, loss_mean_expected.to(loss.dtype))
                loss.backward()
                loss_mean_expected.backward()
                assert torch.allclose(prob.grad, prob_lattice.grad, atol=1e-5)
            else:
                assert torch.allclose(loss.values,
                                      prob_normalized.to(loss.dtype))
                loss_per_utt = loss.sum()
                # The test lattice contains two utts, see `_common_test_part`.
                assert loss_per_utt.shape[0] == 2
                loss_sum = loss_per_utt.sum()
                assert torch.isclose(loss_sum,
                                     loss_sum_expected.to(loss.dtype))
                loss_sum.backward()
                loss_sum_expected.backward()
                assert torch.allclose(prob.grad, prob_lattice.grad, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
