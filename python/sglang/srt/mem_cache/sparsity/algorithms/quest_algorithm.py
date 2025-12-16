# SPDX-License-Identifier: Apache-2.0
import logging

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
	BaseSparseAlgorithmImpl,
)

logger = logging.getLogger(__name__)


class QuestAlgorithm(BaseSparseAlgorithmImpl):
	"""
	Query-aware sparse attention using range bounds (Quest).

	Maintains per-page key minima and maxima to bound query-key dot products.
	The upper bound acts as an approximated criticality score for TopK page
	selection while retaining recent pages via the base implementation.
	"""

	def __init__(self, config, device: torch.device, **kwargs):
		super().__init__(config, device, **kwargs)
		self.page_key_min = {}
		self.page_key_max = {}

	def _initialize_representation_pools(
		self, start_layer: int, end_layer: int, total_num_pages: int
	):
		key_buffer = self.token_to_kv_pool.get_key_buffer(start_layer)
		key_shape = tuple(key_buffer.shape[1:])
		dtype = key_buffer.dtype

		for layer_id in range(start_layer, end_layer):
			self.page_key_min[layer_id] = torch.full(
				(total_num_pages,) + key_shape,
				float("inf"),
				dtype=dtype,
				device=self.device,
			)
			self.page_key_max[layer_id] = torch.full(
				(total_num_pages,) + key_shape,
				float("-inf"),
				dtype=dtype,
				device=self.device,
			)

		logger.info(
			f"Initialized Quest representation pools: {total_num_pages} pages, "
			f"{end_layer - start_layer} layers"
		)

	def _compute_page_representations(
		self,
		layer_id: int,
		reqs: torch.Tensor,
		seq_lens: torch.Tensor,
		start_page,
		end_page: torch.Tensor,
		k_buffer: torch.Tensor,
	):
		if isinstance(start_page, int):
			start_page = torch.full_like(end_page, start_page)

		device = k_buffer.device
		req_to_token = self.req_to_token_pool.req_to_token
		n = reqs.shape[0]
		max_pages = int((end_page - start_page).max().item())
		if max_pages <= 0:
			return

		page_offset = torch.arange(max_pages, device=device).unsqueeze(0)
		page_ids = start_page.unsqueeze(1) + page_offset
		page_mask = page_ids < end_page.unsqueeze(1)

		tok_start = page_ids * self.page_size
		tok_offset = torch.arange(self.page_size, device=device).view(1, 1, -1)
		tok_pos = tok_start.unsqueeze(2) + tok_offset
		tok_mask = (
			tok_pos
			< (tok_start + self.page_size)
			.clamp(max=seq_lens.unsqueeze(1))
			.unsqueeze(2)
		) & page_mask.unsqueeze(2)

		phys_tok = req_to_token[
			reqs.view(n, 1, 1).expand(n, max_pages, self.page_size),
			tok_pos.clamp(0, req_to_token.shape[1] - 1),
		].clamp(0, k_buffer.shape[0] - 1)

		keys = k_buffer[phys_tok]

		# Expand mask to key shape for elementwise min/max
		key_mask = tok_mask
		while key_mask.dim() < keys.dim():
			key_mask = key_mask.unsqueeze(-1)

		pos_inf = torch.tensor(float("inf"), device=device, dtype=keys.dtype)
		neg_inf = torch.tensor(float("-inf"), device=device, dtype=keys.dtype)

		page_min = torch.where(key_mask, keys, pos_inf).amin(dim=2)
		page_max = torch.where(key_mask, keys, neg_inf).amax(dim=2)

		phys_pg = (
			req_to_token[
				reqs.unsqueeze(1).expand(n, max_pages),
				tok_start.clamp(0, req_to_token.shape[1] - 1),
			]
			// self.page_size
		)

		idx = page_mask.nonzero(as_tuple=False)
		if idx.numel() == 0:
			return

		target_min = self.page_key_min[layer_id]
		target_max = self.page_key_max[layer_id]

		selected_pg = phys_pg[idx[:, 0], idx[:, 1]]
		target_min[selected_pg] = page_min[idx[:, 0], idx[:, 1]].to(target_min.dtype)
		target_max[selected_pg] = page_max[idx[:, 0], idx[:, 1]].to(target_max.dtype)

	def _retrieve_page_scores(
		self,
		layer_id: int,
		phys_pages: torch.Tensor,
		req_pool_indices: torch.Tensor,
		queries: torch.Tensor,
	) -> torch.Tensor:
		key_min = self.page_key_min[layer_id]
		key_max = self.page_key_max[layer_id]

		phys_pages_clamped = phys_pages.clamp(0, key_min.shape[0] - 1)
		kmin = key_min[phys_pages_clamped].to(torch.float32)
		kmax = key_max[phys_pages_clamped].to(torch.float32)

		q = queries.to(torch.float32)
		head_dim = kmin.shape[-1]
		bs = q.shape[0]

		if q.dim() == 3:
			if q.shape[-1] == head_dim:
				q_heads = q
			elif q.shape[-1] % head_dim == 0:
				q_heads = q.view(bs, -1, head_dim)
			else:
				raise ValueError(
					f"Query last dim {q.shape[-1]} not compatible with head dim {head_dim}"
				)
		elif q.dim() == 2:
			if q.shape[-1] % head_dim == 0:
				q_heads = q.view(bs, -1, head_dim)
			else:
				raise ValueError(
					f"Query hidden size {q.shape[-1]} not divisible by head dim {head_dim}"
				)
		else:
			raise ValueError(f"Unexpected query shape {tuple(q.shape)} for Quest")

		kv_heads = kmin.shape[-2]
		if q_heads.shape[1] == kv_heads:
			q_grouped = q_heads
		elif q_heads.shape[1] % kv_heads == 0:
			group = q_heads.shape[1] // kv_heads
			q_view = q_heads.view(bs, kv_heads, group, head_dim)
			q_pos = q_view.clamp(min=0).max(dim=2).values
			q_neg = q_view.clamp(max=0).min(dim=2).values
			q_grouped = torch.where(q_pos > 0, q_pos, q_neg)
		else:
			raise ValueError(
				f"Query heads {q_heads.shape[1]} not compatible with KV heads {kv_heads}"
			)

		q = q_grouped.unsqueeze(1)  # [bs, 1, kv_heads, head_dim]

		selected_keys = torch.where(q >= 0, kmax, kmin)
		products = q * selected_keys
		scores = products.sum(dim=(-1, -2))

		kmin_finite = torch.isfinite(kmin.view(kmin.shape[0], kmin.shape[1], -1)).all(
			dim=-1
		)
		kmax_finite = torch.isfinite(kmax.view(kmax.shape[0], kmax.shape[1], -1)).all(
			dim=-1
		)
		valid = kmin_finite & kmax_finite
		scores = torch.where(
			valid, scores, torch.full_like(scores, float("-inf"), device=scores.device)
		)
		print(scores.shape)
		print(scores)
		return scores