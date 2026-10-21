# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable adapter for the installed serving runtime's arithmetic.

Native kernels supply forward values. GDN retains native chunk factors for FLA backward;
decode uses an explicit recurrence adjoint. QDQ uses identity STE.
The optional vLLM dependency supplies kernels; training owns its state and needs no server.
"""

import torch
from torch.autograd.function import once_differentiable


class _Prefill(torch.autograd.Function):
    """Run native prefill once and reuse GDN's prepared operands and chunk factor in backward."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        state,
        scale,
        beta_dtype,
        normalize,
        raw_g,
        raw_beta,
        rate,
        bias,
        precision,
    ):
        # Keep optional inference kernels isolated to the serving profile.
        from ...kernels.quantization.linear_attention.serving.forward import prefill

        q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
        out, final, intermediates = prefill(
            q,
            k,
            v,
            g.float(),
            beta.to(beta_dtype),
            state.float(),
            scale,
            normalize,
            None if raw_g is None else (raw_g, raw_beta, rate, bias),
            precision=precision,
            return_intermediates=True,
        )
        ctx.native_gdn = intermediates is not None
        if ctx.native_gdn:
            prepared_q, prepared_k, g, beta, cumulative, factor = intermediates
            ctx.save_for_backward(
                q,
                k,
                v,
                g[0],
                beta[0],
                state,
                prepared_q,
                prepared_k,
                cumulative,
                factor,
                raw_g,
                raw_beta,
                rate,
                bias,
            )
        else:
            ctx.save_for_backward(q, k, v, g, beta, state)
        ctx.scale, ctx.normalize = scale, normalize
        ctx.set_materialize_grads(False)
        return out.float(), final

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_final):
        # FLA is optional and needed only for prefix gradients, never native inference.
        from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
        from fla.ops.utils import chunk_local_cumsum
        from fla.ops.utils.constant import RCP_LN2

        q, k, v, g, beta, state = (x[None].contiguous() for x in ctx.saved_tensors[:6])
        if ctx.normalize:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        if ctx.native_gdn:
            q, k, cumulative, a, raw_g, raw_beta, rate, bias = ctx.saved_tensors[6:]
            # vLLM stores natural-log cumulative gates; FLA evaluates them with exp2.
            cumulative = cumulative * RCP_LN2
        else:
            cumulative = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
        kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "g": cumulative,
            "beta": beta,
            "scale": ctx.scale,
            "initial_state": state,
            "do": torch.zeros_like(v)
            if grad_output is None
            else grad_output[None].to(v).contiguous(),
            "dht": None if grad_final is None else grad_final[None].contiguous(),
        }
        # Rebuild factors only when the native forward did not retain them.
        # Upstream backward reconstructs chunk states without a second full prefill.
        if g.ndim == 4:
            from fla.ops.kda.chunk_bwd import chunk_kda_bwd
            from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra

            aqk, akk = chunk_kda_fwd_intra(q, k, v, cumulative, beta, ctx.scale)[-2:]
            dq, dk, dv, db, dg, dh, *_ = chunk_kda_bwd(**kwargs, Aqk=aqk, Akk=akk)
        else:
            from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd
            from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra

            if not ctx.native_gdn:
                a = chunk_gated_delta_rule_fwd_intra(k, v, cumulative, beta)[-1]
            dq, dk, dv, db, dg, dh, *_ = chunk_gated_delta_rule_bwd(**kwargs, A=a)
        if ctx.normalize:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        raw_grads = (None,) * 4
        if ctx.native_gdn:
            from fla.ops.common.gate import fused_beta_sigmoid_bwd
            from fla.ops.gated_delta_rule.gate import gdn_gate_bwd

            # Differentiate the raw BF16 projections consumed by native preparation.
            # The caller's independently prepared g/beta did not participate in this forward.
            d_raw_g, d_rate, d_bias = gdn_gate_bwd(
                raw_g.bfloat16().float().contiguous(),
                rate.contiguous(),
                bias.contiguous(),
                dg[0].contiguous(),
            )
            d_raw_beta = fused_beta_sigmoid_bwd(
                raw_beta.bfloat16().float().contiguous(), db[0].contiguous()
            )
            raw_grads = (d_raw_g, d_raw_beta, d_rate, d_bias)
        # Match FLA's BF16 operand-gradient rounding before upstream FP32 casts.
        grads = tuple(
            dx[0].to(x.dtype) for dx, x in zip((dq, dk, dv, dg, db, dh), (q, k, v, g, beta, state))
        )
        if ctx.native_gdn:
            grads = (*grads[:3], None, None, grads[5])
        return (*grads, None, None, None, *raw_grads, None)


def prefix(
    q, k, v, g, beta, state, scale, beta_dtype, normalize=False, gate_inputs=None, precision="vllm"
):
    """Return native prefix output/state with an upstream FLA recurrence backward."""
    return _Prefill.apply(
        q,
        k,
        v,
        g,
        beta,
        state,
        scale,
        beta_dtype,
        normalize,
        *((None,) * 4 if gate_inputs is None else gate_inputs),
        precision,
    )


class _DecodeStep(torch.autograd.Function):
    """Run native decode once and differentiate the recurrence only during backward."""

    @staticmethod
    def forward(ctx, q, k, v, gate, beta, state, scale, normalize, gate_inputs, precision):
        # Autograd Function.forward runs without recording the inference kernel's operations.
        from ...kernels.quantization.linear_attention.serving.forward import step as native_step

        output, final = native_step(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            gate.float(),
            beta.float(),
            state.float(),
            scale,
            normalize,
            gate_inputs,
            precision=precision,
        )
        ctx.save_for_backward(q, k, v, gate, beta, state, final)
        ctx.scale, ctx.normalize = scale, normalize
        ctx.set_materialize_grads(False)
        return output.float(), final

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_final):
        if grad_output is None and grad_final is None:
            return (None,) * 10
        q, k, v, gate, beta, state, final = ctx.saved_tensors
        if ctx.normalize:
            q_norm = (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            k_norm = (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            q, k = q / q_norm, k / k_norm
        decay = gate.exp().unsqueeze(-1)
        if gate.ndim == 1:
            decay = decay.unsqueeze(-1)
        decayed = state * decay
        residual = v - (decayed * k.unsqueeze(-1)).sum(-2)

        # Output reads the native working state before the external QDQ cache write.
        grad_q, grad_working = None, grad_final
        if grad_output is not None:
            grad_q = (final * grad_output.unsqueeze(-2)).sum(-1) * ctx.scale
            readout_grad = (q * ctx.scale).unsqueeze(-1) * grad_output.unsqueeze(-2)
            grad_working = readout_grad if grad_final is None else grad_final + readout_grad
        grad_update = (grad_working * k.unsqueeze(-1)).sum(-2)
        grad_v = grad_update * beta.unsqueeze(-1)
        grad_beta = (grad_update * residual).sum(-1)
        grad_k = (grad_working * (residual * beta.unsqueeze(-1)).unsqueeze(-2)).sum(-1)
        grad_k = grad_k - (decayed * grad_v.unsqueeze(-2)).sum(-1)
        grad_decayed = grad_working - k.unsqueeze(-1) * grad_v.unsqueeze(-2)
        grad_state = grad_decayed * decay
        grad_gate = (grad_decayed * decayed).sum(-1)
        if gate.ndim == 1:
            grad_gate = grad_gate.sum(-1)
        if ctx.normalize:
            if grad_q is not None:
                grad_q = (grad_q - q * (grad_q * q).sum(-1, keepdim=True)) / q_norm
            grad_k = (grad_k - k * (grad_k * k).sum(-1, keepdim=True)) / k_norm
        return grad_q, grad_k, grad_v, grad_gate, grad_beta, grad_state, None, None, None, None


def step(q, k, v, gate, beta, state, scale, normalize=False, gate_inputs=None, precision="vllm"):
    """Return native output/state with a recurrence backward; the caller owns state QDQ."""
    return _DecodeStep.apply(q, k, v, gate, beta, state, scale, normalize, gate_inputs, precision)
