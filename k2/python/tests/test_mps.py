"""Tests for k2 Apple Metal (MPS) backend support."""
import pytest
import torch
import k2

mps_available = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)


@mps_available
class TestMpsContext:
    def test_linear_fsa_to_mps(self):
        fsa = k2.linear_fsa([1, 2, 3])
        mps_fsa = fsa.to('mps')
        assert mps_fsa.device.type == 'mps'

    def test_round_trip(self):
        fsa = k2.linear_fsa([1, 2, 3])
        mps_fsa = fsa.to('mps')
        cpu_fsa = mps_fsa.to('cpu')
        assert (fsa.arcs.values() == cpu_fsa.arcs.values()).all()

    def test_ctc_topo_mps(self):
        fsa = k2.ctc_topo(5)
        mps_fsa = fsa.to('mps')
        assert mps_fsa.device.type == 'mps'
        cpu_back = mps_fsa.to('cpu')
        assert (fsa.arcs.values() == cpu_back.arcs.values()).all()

    def test_arc_sort_mps(self):
        """arc_sort on MPS must produce correctly sorted arcs."""
        # Use an unsorted FSA so we can verify the sort actually ran.
        fsa = k2.linear_fsa([3, 1, 2])
        # arc_sort on CPU gives the reference order.
        sorted_cpu = k2.arc_sort(fsa)
        # Move to MPS and sort there; round-trip back for comparison.
        mps_fsa = fsa.to('mps')
        sorted_mps = k2.arc_sort(mps_fsa)
        assert sorted_mps.device.type == 'mps'
        assert (
            sorted_cpu.arcs.values()
            == sorted_mps.to('cpu').arcs.values()
        ).all()

    def test_ragged_to_mps(self):
        # Includes a single-element row to exercise ExclusiveSum/InclusiveSum
        # n=1.
        ragged = k2.RaggedTensor([[1, 2, 3], [4, 5], [6]])
        mps_ragged = ragged.to(device='mps')
        cpu_back = mps_ragged.to(device='cpu')
        assert (ragged.values == cpu_back.values).all()
        # Verify row structure (row_ids) round-trips correctly.
        assert (ragged.shape.row_ids(1) == cpu_back.shape.row_ids(1)).all()

    def test_with_mps_flag(self):
        """k2.with_mps must be True when built with MPS support."""
        assert k2.with_mps, "k2 was not built with MPS support"


@mps_available
class TestMpsTraining:
    def test_mutual_information_backward_mps(self):
        B, S, T = 2, 4, 5
        px = torch.randn(B, S, T + 1, device='mps', requires_grad=True)
        py = torch.randn(B, S + 1, T, device='mps', requires_grad=True)
        mi = k2.mutual_information_recursion(px, py)
        mi.sum().backward()
        assert px.grad is not None
        assert px.grad.device.type == 'mps'
        assert py.grad is not None
        assert py.grad.device.type == 'mps'

    def test_mutual_information_parity(self):
        """MPS result must match CPU to within tolerance."""
        B, S, T = 2, 4, 5
        px_cpu = torch.randn(B, S, T + 1)
        py_cpu = torch.randn(B, S + 1, T)
        mi_cpu = k2.mutual_information_recursion(px_cpu, py_cpu)

        px_mps = px_cpu.to('mps')
        py_mps = py_cpu.to('mps')
        mi_mps = k2.mutual_information_recursion(px_mps, py_mps)

        assert torch.allclose(mi_cpu, mi_mps.cpu(), atol=1e-4)

    def test_mutual_information_with_boundary_mps(self):
        """mutual_information_recursion with boundary tensor on MPS."""
        B, S, T = 2, 4, 5
        px_cpu = torch.randn(B, S, T + 1)
        py_cpu = torch.randn(B, S + 1, T)
        boundary = torch.zeros(B, 4, dtype=torch.int64)
        boundary[:, 2] = S
        boundary[:, 3] = T

        mi_cpu = k2.mutual_information_recursion(px_cpu, py_cpu,
                                                  boundary=boundary)

        px_mps = px_cpu.to('mps')
        py_mps = py_cpu.to('mps')
        boundary_mps = boundary.to('mps')
        mi_mps = k2.mutual_information_recursion(px_mps, py_mps,
                                                  boundary=boundary_mps)

        assert torch.allclose(mi_cpu, mi_mps.cpu(), atol=1e-4)

    def test_mps_scores_bridge(self):
        """_MpsScoresBridge: forward→CPU, backward→MPS with correct grads."""
        mps_t = torch.randn(4, device='mps', requires_grad=True)
        cpu_t = k2.autograd._MpsScoresBridge.apply(mps_t)
        assert cpu_t.device.type == 'cpu'
        cpu_t.sum().backward()
        assert mps_t.grad is not None
        assert mps_t.grad.device.type == 'mps'
        # d(sum)/d(x_i) = 1 for all i — verify gradient magnitudes are correct.
        assert torch.allclose(mps_t.grad.cpu(), torch.ones(4))

    def test_tot_scores_log_backward_mps(self):
        """log-semiring get_tot_scores backward: grads on MPS, match CPU."""
        fsa_a = k2.linear_fsa([1, 2])
        fsa_b = k2.linear_fsa([3])
        fsa_vec = k2.create_fsa_vec([fsa_a, fsa_b])

        # CPU reference
        cpu_fsa = fsa_vec.clone()
        cpu_fsa.scores.requires_grad_(True)
        cpu_tot = cpu_fsa.get_tot_scores(log_semiring=True,
                                          use_double_scores=False)
        cpu_tot.sum().backward()

        # MPS under test
        mps_fsa = fsa_vec.to('mps')
        mps_fsa.scores.requires_grad_(True)
        mps_tot = mps_fsa.get_tot_scores(log_semiring=True,
                                          use_double_scores=False)
        mps_tot.sum().backward()

        assert mps_fsa.scores.grad is not None
        assert mps_fsa.scores.grad.device.type == 'mps'
        assert torch.allclose(cpu_fsa.scores.grad,
                               mps_fsa.scores.grad.cpu(), atol=1e-5)

    def test_tot_scores_tropical_backward_mps(self):
        """tropical-semiring get_tot_scores backward: grads land on MPS."""
        fsa_a = k2.linear_fsa([1, 2])
        fsa_b = k2.linear_fsa([3])
        fsa_vec = k2.create_fsa_vec([fsa_a, fsa_b])

        cpu_fsa = fsa_vec.clone()
        cpu_fsa.scores.requires_grad_(True)
        cpu_tot = cpu_fsa.get_tot_scores(log_semiring=False,
                                          use_double_scores=False)
        cpu_tot.sum().backward()

        mps_fsa = fsa_vec.to('mps')
        mps_fsa.scores.requires_grad_(True)
        mps_tot = mps_fsa.get_tot_scores(log_semiring=False,
                                          use_double_scores=False)
        mps_tot.sum().backward()

        assert mps_fsa.scores.grad is not None
        assert mps_fsa.scores.grad.device.type == 'mps'
        assert torch.allclose(cpu_fsa.scores.grad,
                               mps_fsa.scores.grad.cpu(), atol=1e-5)

    def test_tot_scores_double_mps(self):
        """use_double_scores=True on MPS: result as float32 (no float64)."""
        fsa = k2.linear_fsa([1, 2])
        fsa_vec = k2.create_fsa_vec([fsa])

        cpu_tot = fsa_vec.get_tot_scores(log_semiring=True,
                                          use_double_scores=True)

        mps_fsa = fsa_vec.to('mps')
        mps_fsa.scores.requires_grad_(True)
        mps_tot = mps_fsa.get_tot_scores(log_semiring=True,
                                          use_double_scores=True)

        # MPS has no float64 support; result is downcast to float32.
        assert mps_tot.device.type == 'mps'
        assert mps_tot.dtype == torch.float32
        assert torch.allclose(cpu_tot.float(), mps_tot.cpu(), atol=1e-5)

        # Gradients must flow back to MPS scores even with the float64 downcast.
        mps_tot.sum().backward()
        assert mps_fsa.scores.grad is not None
        assert mps_fsa.scores.grad.device.type == 'mps'

    def test_tot_scores_nonunit_gradient_mps(self):
        """Non-unit incoming gradient scales score grads correctly on MPS."""
        fsa_a = k2.linear_fsa([1, 2])
        fsa_b = k2.linear_fsa([3])
        fsa_vec = k2.create_fsa_vec([fsa_a, fsa_b])

        # CPU reference with weighted upstream gradient.
        cpu_fsa = fsa_vec.clone()
        cpu_fsa.scores.requires_grad_(True)
        cpu_tot = cpu_fsa.get_tot_scores(log_semiring=True,
                                          use_double_scores=False)
        upstream = torch.tensor([2.0, 0.5])
        cpu_tot.backward(upstream)

        # MPS under test.
        mps_fsa = fsa_vec.to('mps')
        mps_fsa.scores.requires_grad_(True)
        mps_tot = mps_fsa.get_tot_scores(log_semiring=True,
                                          use_double_scores=False)
        mps_tot.backward(upstream.to('mps'))

        assert mps_fsa.scores.grad is not None
        assert mps_fsa.scores.grad.device.type == 'mps'
        assert torch.allclose(cpu_fsa.scores.grad,
                               mps_fsa.scores.grad.cpu(), atol=1e-5)


@mps_available
class TestMpsForwardScores:
    """Tests for native Metal GetForwardScores (Priority 3)."""

    def _make_fsa_vec(self, fsa_str: str, device: str):
        fsa = k2.Fsa.from_str(fsa_str)
        return k2.create_fsa_vec([fsa]).to(device)

    def test_forward_scores_log_parity(self):
        """Metal log-semiring forward scores must match CPU."""
        s = '''
            0 1 0 0.1
            0 1 1 0.2
            1 2 -1 0.3
            2
        '''
        fsa_mps = self._make_fsa_vec(s, 'mps')
        fsa_cpu = self._make_fsa_vec(s, 'cpu')
        fwd_mps = fsa_mps._get_forward_scores(use_double_scores=False,
                                               log_semiring=True)
        fwd_cpu = fsa_cpu._get_forward_scores(use_double_scores=False,
                                               log_semiring=True)
        assert fwd_mps.device.type == 'mps'
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)

    def test_forward_scores_tropical_parity(self):
        """Metal tropical-semiring forward scores must match CPU."""
        s = '''
            0 1 0 0.1
            0 1 1 0.2
            1 2 -1 0.3
            2
        '''
        fsa_mps = self._make_fsa_vec(s, 'mps')
        fsa_cpu = self._make_fsa_vec(s, 'cpu')
        fwd_mps = fsa_mps._get_forward_scores(use_double_scores=False,
                                               log_semiring=False)
        fwd_cpu = fsa_cpu._get_forward_scores(use_double_scores=False,
                                               log_semiring=False)
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)

    def test_forward_scores_differentiable(self):
        """Differentiable get_forward_scores must compute correct gradients."""
        s = '''
            0 1 0 0.1
            0 1 1 0.2
            1 2 -1 0.3
            2
        '''
        fsa_mps = self._make_fsa_vec(s, 'mps')
        fsa_mps.scores.requires_grad_(True)
        fsa_cpu = self._make_fsa_vec(s, 'cpu')
        fsa_cpu.scores.requires_grad_(True)

        fwd_mps = fsa_mps.get_forward_scores(use_double_scores=False,
                                              log_semiring=True)
        fwd_cpu = fsa_cpu.get_forward_scores(use_double_scores=False,
                                             log_semiring=True)
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)

        scale = torch.arange(fwd_mps.numel()).float()
        (scale.to('mps') * fwd_mps).sum().backward()
        (scale * fwd_cpu).sum().backward()
        assert torch.allclose(fsa_mps.scores.grad.cpu(),
                               fsa_cpu.scores.grad, atol=1e-5)

    def test_forward_scores_multi_fsa(self):
        """Metal kernel must handle batched FSAs correctly."""
        s1 = '''
            0 1 0 0.5
            1 2 -1 1.0
            2
        '''
        s2 = '''
            0 1 0 0.1
            0 1 1 0.2
            1 2 -1 0.3
            2
        '''
        fsa_vec_mps = k2.create_fsa_vec(
            [k2.Fsa.from_str(s1), k2.Fsa.from_str(s2)]).to('mps')
        fsa_vec_cpu = k2.create_fsa_vec(
            [k2.Fsa.from_str(s1), k2.Fsa.from_str(s2)])
        fwd_mps = fsa_vec_mps._get_forward_scores(use_double_scores=False,
                                                   log_semiring=True)
        fwd_cpu = fsa_vec_cpu._get_forward_scores(use_double_scores=False,
                                                   log_semiring=True)
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)


@mps_available
class TestMpsIntersectDense:
    """Tests for intersect_dense and intersect_dense_pruned on MPS."""

    # Simple decoding graph shared across tests.
    _FSA_STR = '''
        0 1 1 1.0
        1 2 2 2.0
        2 3 -1 3.0
        3
    '''

    def _make_fsa_vec(self, device):
        # Build on CPU first — create_fsa_vec calls Fsa() which accesses
        # `properties` (K2_EVAL), which crashes when the FSA is on MPS.
        fsa = k2.Fsa.from_str(self._FSA_STR)
        fsa_vec = k2.create_fsa_vec([fsa]).to(device)
        fsa_vec.scores.requires_grad_(True)
        return fsa_vec

    def _make_dense(self, device):
        # DenseFsaVec.__init__ calls _k2.DenseFsaVec(scores, row_splits) where
        # row_splits is read via data_ptr() — crashes on MPS.  Build on CPU
        # then move to device; set requires_grad on the device-resident scores.
        log_prob_cpu = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.04, 0.05, 0.06]]], dtype=torch.float32)
        segs = torch.tensor([[0, 0, 2]], dtype=torch.int32)
        dense = k2.DenseFsaVec(log_prob_cpu, segs)
        if device != 'cpu':
            dense = dense.to(device)
        dense.scores.requires_grad_(True)
        return dense

    def test_intersect_dense_forward_parity(self):
        """intersect_dense on MPS must produce scores matching CPU."""
        fsa_cpu = self._make_fsa_vec('cpu')
        dense_cpu = self._make_dense('cpu')
        out_cpu = k2.intersect_dense(fsa_cpu, dense_cpu, output_beam=100000)

        fsa_mps = self._make_fsa_vec('mps')
        dense_mps = self._make_dense('mps')
        out_mps = k2.intersect_dense(fsa_mps, dense_mps, output_beam=100000)

        assert out_mps.device.type == 'mps'
        assert torch.allclose(out_mps.scores.cpu(), out_cpu.scores, atol=1e-5)

    def test_intersect_dense_backward_parity(self):
        """intersect_dense backward on MPS: grads land on MPS, match CPU."""
        fsa_cpu = self._make_fsa_vec('cpu')
        dense_cpu = self._make_dense('cpu')
        out_cpu = k2.intersect_dense(fsa_cpu, dense_cpu, output_beam=100000)
        out_cpu.get_tot_scores(log_semiring=False,
                               use_double_scores=False).sum().backward()

        fsa_mps = self._make_fsa_vec('mps')
        dense_mps = self._make_dense('mps')
        out_mps = k2.intersect_dense(fsa_mps, dense_mps, output_beam=100000)
        out_mps.get_tot_scores(log_semiring=False,
                               use_double_scores=False).sum().backward()

        # Graph-arc score gradients.
        assert fsa_mps.scores.grad is not None
        assert fsa_mps.scores.grad.device.type == 'mps'
        assert torch.allclose(fsa_mps.scores.grad.cpu(),
                               fsa_cpu.scores.grad, atol=1e-5)
        # Acoustic (b_fsa) score gradients.
        assert dense_mps.scores.grad is not None
        assert dense_mps.scores.grad.device.type == 'mps'
        assert torch.allclose(dense_mps.scores.grad.cpu(),
                               dense_cpu.scores.grad, atol=1e-5)

    def test_intersect_dense_seqframe_attr(self):
        """seqframe_idx_name and frame_idx_name attributes must work on MPS."""
        fsa_mps = self._make_fsa_vec('mps')
        dense_mps = self._make_dense('mps')
        out_mps = k2.intersect_dense(fsa_mps, dense_mps, output_beam=100000,
                                     seqframe_idx_name='seqframe',
                                     frame_idx_name='frame')
        assert hasattr(out_mps, 'seqframe')
        assert hasattr(out_mps, 'frame')
        # Verify against CPU reference.
        fsa_cpu = self._make_fsa_vec('cpu')
        dense_cpu = self._make_dense('cpu')
        out_cpu = k2.intersect_dense(fsa_cpu, dense_cpu, output_beam=100000,
                                     seqframe_idx_name='seqframe',
                                     frame_idx_name='frame')
        assert torch.equal(out_mps.seqframe.cpu(), out_cpu.seqframe)
        assert torch.equal(out_mps.frame.cpu(), out_cpu.frame)

    def test_intersect_dense_pruned_forward_parity(self):
        """intersect_dense_pruned on MPS forward scores must match CPU."""
        fsa_cpu = self._make_fsa_vec('cpu')
        dense_cpu = self._make_dense('cpu')
        out_cpu = k2.intersect_dense_pruned(fsa_cpu, dense_cpu,
                                            search_beam=100000,
                                            output_beam=100000,
                                            min_active_states=0,
                                            max_active_states=10000)

        fsa_mps = self._make_fsa_vec('mps')
        dense_mps = self._make_dense('mps')
        out_mps = k2.intersect_dense_pruned(fsa_mps, dense_mps,
                                            search_beam=100000,
                                            output_beam=100000,
                                            min_active_states=0,
                                            max_active_states=10000)

        assert out_mps.device.type == 'mps'
        assert torch.allclose(out_mps.scores.cpu(), out_cpu.scores, atol=1e-5)

    def test_intersect_dense_pruned_backward_parity(self):
        """intersect_dense_pruned backward on MPS: grads land on MPS."""
        fsa_cpu = self._make_fsa_vec('cpu')
        dense_cpu = self._make_dense('cpu')
        out_cpu = k2.intersect_dense_pruned(fsa_cpu, dense_cpu,
                                            search_beam=100000,
                                            output_beam=100000,
                                            min_active_states=0,
                                            max_active_states=10000)
        out_cpu.get_tot_scores(log_semiring=False,
                               use_double_scores=False).sum().backward()

        fsa_mps = self._make_fsa_vec('mps')
        dense_mps = self._make_dense('mps')
        out_mps = k2.intersect_dense_pruned(fsa_mps, dense_mps,
                                            search_beam=100000,
                                            output_beam=100000,
                                            min_active_states=0,
                                            max_active_states=10000)
        out_mps.get_tot_scores(log_semiring=False,
                               use_double_scores=False).sum().backward()

        assert fsa_mps.scores.grad is not None
        assert fsa_mps.scores.grad.device.type == 'mps'
        assert torch.allclose(fsa_mps.scores.grad.cpu(),
                               fsa_cpu.scores.grad, atol=1e-5)
        assert dense_mps.scores.grad is not None
        assert dense_mps.scores.grad.device.type == 'mps'
        assert torch.allclose(dense_mps.scores.grad.cpu(),
                               dense_cpu.scores.grad, atol=1e-5)


@mps_available
class TestMpsAssocScan:
    """Tests for Priority 6: Hillis-Steele associative-scan forward scores."""

    # 4-state linear-chain FSA: 3 arcs, 4 states → assoc-scan threshold met.
    _LINEAR_STR = '''
        0 1 1 0.5
        1 2 2 1.0
        2 3 -1 1.5
        3
    '''

    # 8-state FSA with branching to exercise prefix products more thoroughly.
    _BRANCHING_STR = '''
        0 1 1 1.0
        0 2 2 2.0
        1 3 3 0.5
        2 3 3 1.5
        3 4 4 0.25
        4 5 5 0.75
        5 6 6 0.5
        6 7 -1 1.0
        7
    '''

    def _make_fsa(self, fsa_str, device='cpu'):
        fsa = k2.Fsa.from_str(fsa_str.strip())
        return k2.create_fsa_vec([fsa]).to(device)

    def test_assoc_scan_linear_tropical_parity(self):
        """Single-FSA tropical forward scores via assoc scan match CPU."""
        fsa_cpu = self._make_fsa(self._LINEAR_STR)
        fsa_mps = self._make_fsa(self._LINEAR_STR, 'mps')

        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=False)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=False)

        assert fwd_mps.device.type == 'mps'
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)

    def test_assoc_scan_linear_log_parity(self):
        """Log-semiring falls back to native path; results still match CPU."""
        fsa_cpu = self._make_fsa(self._LINEAR_STR)
        fsa_mps = self._make_fsa(self._LINEAR_STR, 'mps')

        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=True)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=True)

        assert fwd_mps.device.type == 'mps'
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)

    def test_assoc_scan_branching_tropical_parity(self):
        """Branching 8-state FSA tropical forward scores match CPU."""
        fsa_cpu = self._make_fsa(self._BRANCHING_STR)
        fsa_mps = self._make_fsa(self._BRANCHING_STR, 'mps')

        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=False)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=False)

        assert fwd_mps.device.type == 'mps'
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)

    def test_assoc_scan_tot_scores_parity(self):
        """Total (best-path) scores via assoc-scan forward path match CPU.

        Uses the differentiable get_tot_scores which bridges MPS→CPU for the
        final C++ score extraction (the non-differentiable _get_tot_scores is
        not MPS-safe, as its C++ path reads MPS arcs via K2_EVAL).
        """
        fsa_cpu = self._make_fsa(self._BRANCHING_STR)
        fsa_mps = self._make_fsa(self._BRANCHING_STR, 'mps')

        tot_cpu = fsa_cpu.get_tot_scores(
            use_double_scores=False, log_semiring=False)
        tot_mps = fsa_mps.get_tot_scores(
            use_double_scores=False, log_semiring=False)

        assert tot_mps.device.type == 'mps'
        assert torch.allclose(tot_mps.cpu(), tot_cpu, atol=1e-5)

    def test_assoc_scan_large_fallback(self):
        """FSA with >128 states falls back to native sequential path."""
        # Build a long linear chain with 200 states: exceeds N_MAX=128.
        # The last arc must use label -1 (final/epsilon in k2 convention).
        arcs = []
        for i in range(198):
            arcs.append(f'{i} {i + 1} {i % 100 + 1} 0.01')
        arcs.append('198 199 -1 0.01')
        arcs.append('199')
        fsa_str = '\n'.join(arcs)
        fsa_cpu = self._make_fsa(fsa_str)
        fsa_mps = self._make_fsa(fsa_str, 'mps')

        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=False)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=False)

        assert fwd_mps.device.type == 'mps'
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-4)


@mps_available
class TestMpsNumericalParity:
    """Verify MPS results match CPU within tolerance."""

    def test_linear_fsa_scores_parity(self):
        """Round-tripping non-trivial scores through MPS preserves values."""
        fsa = k2.linear_fsa([1, 2, 3])
        fsa_vec = k2.create_fsa_vec([fsa])
        # Assign non-trivial scores so a no-op copy would be detected.
        fsa_vec.scores = torch.tensor([1.5, -0.5, 2.0, 0.0])
        mps_fsa = fsa_vec.to('mps')
        assert torch.allclose(fsa_vec.scores,
                               mps_fsa.scores.cpu(), atol=1e-6)

    def test_dense_fsa_vec_mps(self):
        """DenseFsaVec.to('mps') round-trip must preserve score values."""
        # Build a small DenseFsaVec on CPU.
        T, num_classes = 5, 4
        log_probs = torch.randn(1, T, num_classes)
        supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32)
        dense = k2.DenseFsaVec(log_probs, supervision_segments)
        mps_dense = dense.to('mps')
        assert mps_dense.scores.device.type == 'mps'
        # Round-trip back and verify values.
        cpu_back = mps_dense.to('cpu')
        assert torch.allclose(dense.scores, cpu_back.scores, atol=1e-6)

# =============================================================================
# PR Audit — Extended Test Suite
# =============================================================================


def _make_fsa_vec(fsa_str, device='cpu'):
    """Helper: build FsaVec from multi-line FSA string, move to device."""
    fsa = k2.Fsa.from_str(fsa_str.strip())
    return k2.create_fsa_vec([fsa]).to(device)


@mps_available
class TestMpsEdgeCases:
    """Edge-case tests: empty FSA, unreachable states, guard paths."""

    def test_empty_arcs_forward_scores(self):
        """FSA with only start/accept state (0 arcs) returns correct scores."""
        # A single-arc FSA with just a final arc to the accept state.
        # This has 2 states: state 0 (start) and state 1 (accept).
        fsa_str = '0 1 -1 0.0\n1'
        fsa_cpu = _make_fsa_vec(fsa_str)
        fsa_mps = _make_fsa_vec(fsa_str, 'mps')

        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=False)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=False)

        assert fwd_mps.device.type == 'mps'
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)

    def test_unreachable_state_forward_scores(self):
        """State with no entering arcs gets -inf forward score."""
        # State 1 is unreachable: arcs go 0→2 and 2→3. State 1 is dangling.
        # k2 requires valid topological structure; use isolated final state.
        fsa_str = '''
            0 2 1 1.0
            0 2 2 2.0
            2 3 -1 0.5
            3
        '''
        fsa_cpu = _make_fsa_vec(fsa_str)
        fsa_mps = _make_fsa_vec(fsa_str, 'mps')

        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=False)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=False)

        assert fwd_mps.device.type == 'mps'
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)

    def test_forward_scores_double_mps_raises(self):
        """_get_forward_scores with use_double_scores=True raises error."""
        fsa_mps = _make_fsa_vec('0 1 -1 1.0\n1', 'mps')
        with pytest.raises(NotImplementedError, match='use_double_scores'):
            fsa_mps._get_forward_scores(
                use_double_scores=True, log_semiring=False)

    def test_backward_scores_mps_raises(self):
        """_get_backward_scores on MPS raises NotImplementedError."""
        fsa_mps = _make_fsa_vec('0 1 -1 1.0\n1', 'mps')
        with pytest.raises(NotImplementedError, match='_get_backward_scores'):
            fsa_mps._get_backward_scores(
                use_double_scores=False, log_semiring=False)

    def test_single_path_forward_scores(self):
        """Single-path chain: MPS forward scores equal manual computation."""
        # 0→1 (w=1.0) → 2 (w=2.0) → 3 (w=3.0, final)
        fsa_str = '0 1 1 1.0\n1 2 2 2.0\n2 3 -1 3.0\n3'
        fsa_mps = _make_fsa_vec(fsa_str, 'mps')
        fwd = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=False)
        # Expected: state 0=0, state 1=1, state 2=3, state 3=6
        expected = torch.tensor([0.0, 1.0, 3.0, 6.0])
        assert torch.allclose(fwd.cpu(), expected, atol=1e-5)

    def test_parallel_arcs_max_score(self):
        """Parallel arcs same src→dst: tropical forward takes maximum."""
        # Two arcs from 0→1: weights 3.0 and 5.0. Max wins.
        fsa_str = '0 1 1 3.0\n0 1 2 5.0\n1 2 -1 1.0\n2'
        fsa_cpu = _make_fsa_vec(fsa_str)
        fsa_mps = _make_fsa_vec(fsa_str, 'mps')

        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=False)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=False)

        assert fwd_mps.device.type == 'mps'
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)
        # State 1 score = max(3.0, 5.0) = 5.0
        assert abs(fwd_mps[1].item() - 5.0) < 1e-5

    def test_parallel_arcs_log_semiring(self):
        """Multiple arcs with same src→dst: log-semiring sums contributions."""
        fsa_str = '0 1 1 1.0\n0 1 2 2.0\n1 2 -1 0.0\n2'
        fsa_cpu = _make_fsa_vec(fsa_str)
        fsa_mps = _make_fsa_vec(fsa_str, 'mps')

        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=True)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=True)

        assert fwd_mps.device.type == 'mps'
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)


@mps_available
class TestMpsAssocScanBoundaries:
    """Boundary conditions for the Priority-6 Hillis-Steele associative scan."""

    def _chain_fsa(self, n_states, device='cpu'):
        """Build a linear-chain FsaVec with n_states states."""
        arcs = [f'{i} {i + 1} {i + 1} {float(i + 1) * 0.1:.1f}'
                for i in range(n_states - 2)]
        arcs.append(
            f'{n_states - 2} {n_states - 1} -1 '
            f'{float(n_states - 1) * 0.1:.1f}'
        )
        arcs.append(str(n_states - 1))
        fsa = k2.Fsa.from_str('\n'.join(arcs).strip())
        return k2.create_fsa_vec([fsa]).to(device)

    def _parity(self, n_states):
        """Assert MPS tropical forward scores match CPU for chain FSA."""
        fsa_cpu = self._chain_fsa(n_states)
        fsa_mps = self._chain_fsa(n_states, 'mps')
        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=False)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=False)
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-4)

    def test_n_at_lower_bound(self):
        """N=4: minimum threshold — uses assoc scan, not native sequential."""
        self._parity(4)

    def test_n_just_above_lower(self):
        """N=5: T_pow2=8, exercises identity-padding in Hillis-Steele."""
        self._parity(5)

    def test_n_nonpower_of_two_small(self):
        """N=7: T_pow2=8, two extra identity padding matrices."""
        self._parity(7)

    def test_n_at_power_of_two_mid(self):
        """N=16: T_pow2=16, no padding needed."""
        self._parity(16)

    def test_n_at_upper_bound(self):
        """N=128: maximum threshold — still uses assoc scan."""
        self._parity(128)

    def test_n_just_above_upper(self):
        """N=129: above threshold — falls back to native sequential."""
        self._parity(129)

    def test_diamond_topology_assoc_scan(self):
        """Diamond: two paths to same dest; assoc scan atomic-max is correct."""
        # 0 → 1 (w=1.0), 0 → 2 (w=2.0), 1 → 3 (w=1.5), 2 → 3 (w=0.5), 3 final
        fsa_str = '''
            0 1 1 1.0
            0 2 2 2.0
            1 3 3 1.5
            2 3 3 0.5
            3 4 -1 0.0
            4
        '''
        fsa_cpu = _make_fsa_vec(fsa_str)
        fsa_mps = _make_fsa_vec(fsa_str, 'mps')

        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=False)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=False)

        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)
        # Best path: 0→2→3 = 2.0+0.5=2.5; 0→1→3 = 1.0+1.5=2.5 (tie → max = 2.5)
        assert abs(fwd_mps[3].item() - 2.5) < 1e-5

    def test_multi_arc_single_dest_assoc_scan(self):
        """Multiple arcs into same destination in assoc scan build_level."""
        # State 2: arcs from 0 (w=1.0), 1a (w=3.0), 1b (w=2.0).
        # The build_level kernel's atomic-max must keep 3.0.
        fsa_str = '''
            0 1 1 1.0
            0 2 2 1.0
            1 2 3 3.0
            1 2 4 2.0
            2 3 -1 0.0
            3
        '''
        fsa_cpu = _make_fsa_vec(fsa_str)
        fsa_mps = _make_fsa_vec(fsa_str, 'mps')

        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=False)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=False)

        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)

    def test_assoc_scan_log_semiring_falls_back(self):
        """Log semiring falls back to native; result still matches CPU."""
        fsa_str = '0 1 1 0.5\n1 2 -1 0.5\n2'
        fsa_cpu = _make_fsa_vec(fsa_str)
        fsa_mps = _make_fsa_vec(fsa_str, 'mps')
        fwd_cpu = fsa_cpu._get_forward_scores(
            use_double_scores=False, log_semiring=True)
        fwd_mps = fsa_mps._get_forward_scores(
            use_double_scores=False, log_semiring=True)
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-5)


@mps_available
class TestMpsArcPost:
    """Tests for arc posteriors (differentiable get_arc_post) on MPS."""

    _FSA_STR = '''
        0 1 1 1.0
        0 1 2 2.0
        1 2 -1 0.5
        2
    '''

    def _make(self, device='cpu'):
        fsa = k2.Fsa.from_str(self._FSA_STR.strip())
        fsa_vec = k2.create_fsa_vec([fsa]).to(device)
        fsa_vec.scores.requires_grad_(True)
        return fsa_vec

    def test_arc_post_tropical_parity(self):
        """Tropical arc posteriors on MPS match CPU."""
        fsa_cpu = self._make()
        fsa_mps = self._make('mps')

        post_cpu = fsa_cpu.get_arc_post(
            use_double_scores=False, log_semiring=False)
        post_mps = fsa_mps.get_arc_post(
            use_double_scores=False, log_semiring=False)

        assert post_mps.device.type == 'mps'
        assert torch.allclose(post_mps.cpu(), post_cpu, atol=1e-5)

    def test_arc_post_log_parity(self):
        """Log-semiring arc posteriors on MPS match CPU."""
        fsa_cpu = self._make()
        fsa_mps = self._make('mps')

        post_cpu = fsa_cpu.get_arc_post(
            use_double_scores=False, log_semiring=True)
        post_mps = fsa_mps.get_arc_post(
            use_double_scores=False, log_semiring=True)

        assert post_mps.device.type == 'mps'
        assert torch.allclose(post_mps.cpu(), post_cpu, atol=1e-5)

    def test_arc_post_tropical_gradient(self):
        """Gradients flow correctly through get_arc_post on MPS (tropical)."""
        fsa_cpu = self._make()
        fsa_mps = self._make('mps')

        post_cpu = fsa_cpu.get_arc_post(
            use_double_scores=False, log_semiring=False)
        post_mps = fsa_mps.get_arc_post(
            use_double_scores=False, log_semiring=False)

        post_cpu.sum().backward()
        post_mps.sum().backward()

        assert fsa_mps.scores.grad is not None
        assert fsa_mps.scores.grad.device.type == 'mps'
        assert torch.allclose(
            fsa_mps.scores.grad.cpu(), fsa_cpu.scores.grad, atol=1e-5)

    def test_arc_post_log_gradient(self):
        """Gradients flow correctly through get_arc_post on MPS (log)."""
        fsa_cpu = self._make()
        fsa_mps = self._make('mps')

        post_cpu = fsa_cpu.get_arc_post(
            use_double_scores=False, log_semiring=True)
        post_mps = fsa_mps.get_arc_post(
            use_double_scores=False, log_semiring=True)

        post_cpu.sum().backward()
        post_mps.sum().backward()

        assert fsa_mps.scores.grad is not None
        assert fsa_mps.scores.grad.device.type == 'mps'
        assert torch.allclose(
            fsa_mps.scores.grad.cpu(), fsa_cpu.scores.grad, atol=1e-5)


@mps_available
class TestMpsGetForwardScoresDifferentiable:
    """Differentiable get_forward_scores tests on MPS."""

    _FSA_STR = '''
        0 1 1 0.5
        0 2 2 1.0
        1 3 3 0.5
        2 3 3 1.0
        3 4 -1 0.0
        4
    '''

    def _make(self, device='cpu'):
        fsa = k2.Fsa.from_str(self._FSA_STR.strip())
        fsa_vec = k2.create_fsa_vec([fsa]).to(device)
        fsa_vec.scores.requires_grad_(True)
        return fsa_vec

    def test_get_forward_scores_tropical_gradient(self):
        """Differentiable tropical forward scores: grad on MPS matches CPU."""
        fsa_cpu = self._make()
        fsa_mps = self._make('mps')

        fwd_cpu = fsa_cpu.get_forward_scores(
            use_double_scores=False, log_semiring=False)
        fwd_mps = fsa_mps.get_forward_scores(
            use_double_scores=False, log_semiring=False)

        fwd_cpu.sum().backward()
        fwd_mps.sum().backward()

        assert fsa_mps.scores.grad.device.type == 'mps'
        assert torch.allclose(
            fsa_mps.scores.grad.cpu(), fsa_cpu.scores.grad, atol=1e-5)

    def test_get_forward_scores_log_gradient(self):
        """Differentiable log-semiring forward scores: MPS grad matches CPU."""
        fsa_cpu = self._make()
        fsa_mps = self._make('mps')

        fwd_cpu = fsa_cpu.get_forward_scores(
            use_double_scores=False, log_semiring=True)
        fwd_mps = fsa_mps.get_forward_scores(
            use_double_scores=False, log_semiring=True)

        fwd_cpu.sum().backward()
        fwd_mps.sum().backward()

        assert fsa_mps.scores.grad.device.type == 'mps'
        assert torch.allclose(
            fsa_mps.scores.grad.cpu(), fsa_cpu.scores.grad, atol=1e-5)

    def test_get_forward_scores_nonunit_gradient(self):
        """Non-unit upstream gradient correctly scales MPS grad."""
        fsa_cpu = self._make()
        fsa_mps = self._make('mps')

        fwd_cpu = fsa_cpu.get_forward_scores(
            use_double_scores=False, log_semiring=True)
        fwd_mps = fsa_mps.get_forward_scores(
            use_double_scores=False, log_semiring=True)

        upstream = torch.ones_like(fwd_cpu) * 2.0
        fwd_cpu.backward(upstream)
        fwd_mps.backward(upstream.to('mps'))

        assert torch.allclose(
            fsa_mps.scores.grad.cpu(), fsa_cpu.scores.grad, atol=1e-5)


@mps_available
class TestMpsIntersectDenseExtended:
    """Extended IntersectDense/IntersectDensePruned tests (Priority 5)."""

    _FSA_STR = '''
        0 1 1 0.0
        1 2 2 0.0
        2 3 -1 0.0
        3
    '''

    def _make_fsa(self, device='cpu'):
        fsa = k2.Fsa.from_str(self._FSA_STR.strip())
        fsa_vec = k2.create_fsa_vec([fsa]).to(device)
        fsa_vec.scores.requires_grad_(True)
        return fsa_vec

    def _make_dense(self, n_utterances, device='cpu', seed=0):
        """Build a DenseFsaVec with n_utterances independent segments.

        Uses a fixed seed so that paired CPU/MPS calls with the same seed
        produce identical inputs, enabling genuine parity checks.
        """
        torch.manual_seed(seed)
        T, V = 3, 3
        log_probs = torch.randn(n_utterances, T, V)
        segs = torch.tensor(
            [[i, 0, T] for i in range(n_utterances)], dtype=torch.int32)
        dense = k2.DenseFsaVec(log_probs, segs)
        if device != 'cpu':
            dense = dense.to(device)
        dense.scores.requires_grad_(True)
        return dense

    def test_intersect_dense_pruned_2utterances(self):
        """IntersectDensePruned with 2-utterance batch: scores match CPU."""
        fsa_cpu = self._make_fsa()
        fsa_mps = self._make_fsa('mps')
        dense_cpu = self._make_dense(2, seed=7)
        dense_mps = self._make_dense(2, 'mps', seed=7)

        result_cpu = k2.intersect_dense_pruned(
            fsa_cpu, dense_cpu,
            search_beam=20.0, output_beam=8.0,
            min_active_states=30, max_active_states=10000)
        result_mps = k2.intersect_dense_pruned(
            fsa_mps, dense_mps,
            search_beam=20.0, output_beam=8.0,
            min_active_states=30, max_active_states=10000)

        assert result_mps.scores.device.type == 'mps'
        assert torch.allclose(
            result_mps.scores.cpu(), result_cpu.scores, atol=1e-5)

    def test_intersect_dense_pruned_backward_2utterances(self):
        """IntersectDensePruned 2-utterance backward: grads on MPS match CPU."""
        fsa_cpu = self._make_fsa()
        fsa_mps = self._make_fsa('mps')
        dense_cpu = self._make_dense(2, seed=7)
        dense_mps = self._make_dense(2, 'mps', seed=7)

        result_cpu = k2.intersect_dense_pruned(
            fsa_cpu, dense_cpu,
            search_beam=20.0, output_beam=8.0,
            min_active_states=30, max_active_states=10000)
        result_mps = k2.intersect_dense_pruned(
            fsa_mps, dense_mps,
            search_beam=20.0, output_beam=8.0,
            min_active_states=30, max_active_states=10000)

        result_cpu.scores.sum().backward()
        result_mps.scores.sum().backward()

        assert fsa_mps.scores.grad.device.type == 'mps'
        assert torch.allclose(
            fsa_mps.scores.grad.cpu(), fsa_cpu.scores.grad, atol=1e-5)
        assert torch.allclose(
            dense_mps.scores.grad.cpu(), dense_cpu.scores.grad, atol=1e-5)

    def test_intersect_dense_with_seqframe_attribute(self):
        """seqframe attribute is set correctly on MPS output."""
        fsa_mps = self._make_fsa('mps')
        dense_mps = self._make_dense(1, 'mps', seed=3)

        result = k2.intersect_dense_pruned(
            fsa_mps, dense_mps,
            search_beam=20.0, output_beam=8.0,
            min_active_states=30, max_active_states=10000,
            seqframe_idx_name='seqframe_idx')

        assert hasattr(result, 'seqframe_idx')

    def test_intersect_dense_seqframe_parity(self):
        """seqframe/frame attributes match between CPU and MPS paths."""
        fsa_cpu = self._make_fsa()
        fsa_mps = self._make_fsa('mps')
        dense_cpu = self._make_dense(1, seed=3)
        dense_mps = self._make_dense(1, 'mps', seed=3)

        result_cpu = k2.intersect_dense_pruned(
            fsa_cpu, dense_cpu,
            search_beam=20.0, output_beam=8.0,
            min_active_states=30, max_active_states=10000,
            seqframe_idx_name='seqframe_idx',
            frame_idx_name='frame_idx')
        result_mps = k2.intersect_dense_pruned(
            fsa_mps, dense_mps,
            search_beam=20.0, output_beam=8.0,
            min_active_states=30, max_active_states=10000,
            seqframe_idx_name='seqframe_idx',
            frame_idx_name='frame_idx')

        assert torch.equal(
            result_mps.seqframe_idx.cpu(), result_cpu.seqframe_idx)
        assert torch.equal(result_mps.frame_idx.cpu(), result_cpu.frame_idx)

    def test_intersect_dense_function_parity(self):
        """IntersectDense (non-pruned) forward/backward match CPU."""
        fsa_cpu = self._make_fsa()
        fsa_mps = self._make_fsa('mps')
        dense_cpu = self._make_dense(1, seed=5)
        dense_mps = self._make_dense(1, 'mps', seed=5)

        result_cpu = k2.intersect_dense(fsa_cpu, dense_cpu, output_beam=100.0)
        result_mps = k2.intersect_dense(fsa_mps, dense_mps, output_beam=100.0)

        assert torch.allclose(
            result_mps.scores.cpu(), result_cpu.scores, atol=1e-5)

        result_cpu.scores.sum().backward()
        result_mps.scores.sum().backward()

        assert torch.allclose(
            fsa_mps.scores.grad.cpu(), fsa_cpu.scores.grad, atol=1e-5)
        assert torch.allclose(
            dense_mps.scores.grad.cpu(), dense_cpu.scores.grad, atol=1e-5)


@mps_available
class TestMpsMutualInformationExtended:
    """Extended mutual_information tests for MPS (Priority 1)."""

    def test_mutual_information_varied_sizes(self):
        """mutual_information with several (S, T) sizes matches CPU."""
        for S, T in [(2, 3), (5, 10), (10, 5), (20, 30)]:
            px = torch.randn(1, S, T).requires_grad_(True)
            py = torch.randn(1, S + 1, T).requires_grad_(True)
            px_mps = px.detach().to('mps').requires_grad_(True)
            py_mps = py.detach().to('mps').requires_grad_(True)

            mi_cpu = k2.mutual_information_recursion(px, py)
            mi_mps = k2.mutual_information_recursion(px_mps, py_mps)

            assert torch.allclose(mi_mps.cpu(), mi_cpu, atol=1e-4), (
                f"Failed at S={S}, T={T}: "
                f"MPS={mi_mps.item():.4f} CPU={mi_cpu.item():.4f}"
            )

    def test_mutual_information_gradient_varied_sizes(self):
        """mutual_information backward is correct for several (S, T) sizes."""
        # Sizes where S >= T trigger a pre-existing k2 CPU MI backward warning
        # that leaves CPU gradients as zero (upstream issue unrelated to MPS).
        # Only test shapes where S < T to get reliable CPU reference gradients.
        for S, T in [(3, 4), (5, 7), (4, 8)]:
            px = torch.randn(1, S, T).requires_grad_(True)
            py = torch.randn(1, S + 1, T).requires_grad_(True)
            px_mps = px.detach().to('mps').requires_grad_(True)
            py_mps = py.detach().to('mps').requires_grad_(True)

            k2.mutual_information_recursion(px, py).backward()
            k2.mutual_information_recursion(px_mps, py_mps).backward()

            assert torch.allclose(px_mps.grad.cpu(), px.grad, atol=1e-4), \
                f"px.grad mismatch at S={S}, T={T}"
            assert torch.allclose(py_mps.grad.cpu(), py.grad, atol=1e-4), \
                f"py.grad mismatch at S={S}, T={T}"

    def test_mutual_information_batch(self):
        """Batch of B sequences: results match CPU for each item."""
        B, S, T = 4, 5, 8
        px = torch.randn(B, S, T).requires_grad_(True)
        py = torch.randn(B, S + 1, T).requires_grad_(True)
        px_mps = px.detach().to('mps').requires_grad_(True)
        py_mps = py.detach().to('mps').requires_grad_(True)

        mi_cpu = k2.mutual_information_recursion(px, py)
        mi_mps = k2.mutual_information_recursion(px_mps, py_mps)

        assert torch.allclose(mi_mps.cpu(), mi_cpu, atol=1e-4)

        mi_cpu.sum().backward()
        mi_mps.sum().backward()
        assert torch.allclose(px_mps.grad.cpu(), px.grad, atol=1e-4)
        assert torch.allclose(py_mps.grad.cpu(), py.grad, atol=1e-4)

    def test_mutual_information_with_boundary(self):
        """mutual_information with explicit boundary tensor matches CPU."""
        S, T = 6, 8
        px = torch.randn(1, S, T)
        py = torch.randn(1, S + 1, T)
        boundary = torch.tensor([[0, 0, S, T]], dtype=torch.int64)

        mi_cpu = k2.mutual_information_recursion(px, py, boundary=boundary)
        mi_mps = k2.mutual_information_recursion(
            px.to('mps'), py.to('mps'), boundary=boundary.to('mps'))

        assert torch.allclose(mi_mps.cpu(), mi_cpu, atol=1e-4)


@mps_available
class TestMpsForwardScoresNumericalStress:
    """Numerical stress tests: larger FSAs, various topologies, precision."""

    def _random_dag_fsa(self, n_states, n_arcs, seed=42, device='cpu'):
        """Generate a connected random DAG FSA on device.

        A linear backbone guarantees every state is reachable and
        co-reachable; extra skip arcs add branching for stress testing.
        """
        torch.manual_seed(seed)
        arcs = []
        # Backbone: guaranteed linear chain 0→1→…→(n-2)→(n-1, final arc).
        for s in range(n_states - 2):
            w = torch.randn(1).item()
            arcs.append(f'{s} {s + 1} 1 {w:.4f}')
        arcs.append(f'{n_states - 2} {n_states - 1} -1 0.0')
        # Extra skip arcs between non-final states only (keeps validity).
        extra = max(0, n_arcs - (n_states - 1))
        for _ in range(extra):
            src = int(torch.randint(0, n_states - 2, (1,)).item())
            dst = int(torch.randint(src + 1, n_states - 1, (1,)).item())
            w = torch.randn(1).item()
            arcs.append(f'{src} {dst} 1 {w:.4f}')
        # Deduplicate and sort.
        arc_lines = list(dict.fromkeys(a for a in arcs))
        arc_lines.sort(key=lambda a: (int(a.split()[0]), int(a.split()[1])))
        fsa_str = '\n'.join(arc_lines + [str(n_states - 1)])
        fsa = k2.Fsa.from_str(fsa_str)
        return k2.create_fsa_vec([fsa]).to(device)

    def test_medium_fsa_tropical_parity(self):
        """50-state random DAG: MPS tropical forward scores match CPU."""
        fsa_cpu = self._random_dag_fsa(50, 100)
        fsa_mps = self._random_dag_fsa(50, 100, device='mps')
        fwd_cpu = fsa_cpu._get_forward_scores(False, False)
        fwd_mps = fsa_mps._get_forward_scores(False, False)
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-4)

    def test_medium_fsa_log_parity(self):
        """50-state random DAG: MPS log-semiring forward scores match CPU."""
        fsa_cpu = self._random_dag_fsa(50, 100)
        fsa_mps = self._random_dag_fsa(50, 100, device='mps')
        fwd_cpu = fsa_cpu._get_forward_scores(False, True)
        fwd_mps = fsa_mps._get_forward_scores(False, True)
        assert torch.allclose(fwd_mps.cpu(), fwd_cpu, atol=1e-4)

    def test_gradient_precision_with_repeated_scores(self):
        """Repeated arc scores should not cause NaN in MPS gradients."""
        # All arcs have the same score → softmax-like gradient should sum to 1.
        fsa_str = '0 1 1 0.0\n0 1 2 0.0\n1 2 -1 0.0\n2'
        fsa_cpu = _make_fsa_vec(fsa_str)
        fsa_mps = _make_fsa_vec(fsa_str, 'mps')
        fsa_cpu.scores.requires_grad_(True)
        fsa_mps.scores.requires_grad_(True)

        tot_cpu = fsa_cpu.get_tot_scores(
            use_double_scores=False, log_semiring=True)
        tot_mps = fsa_mps.get_tot_scores(
            use_double_scores=False, log_semiring=True)

        tot_cpu.backward()
        tot_mps.backward()

        grad_mps = fsa_mps.scores.grad.cpu()
        assert not torch.any(torch.isnan(grad_mps)), "NaN in MPS gradient"
        assert torch.allclose(grad_mps, fsa_cpu.scores.grad, atol=1e-5)
