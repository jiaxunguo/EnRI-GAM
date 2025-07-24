import torch
import torch.nn as nn
import torch.nn.functional as F


class BinghamSampler(nn.Module):

    def __init__(self, K=1, n_s=10):  # κ1≤κ2≤κ3=0
        super().__init__()
        self.K = K        
        self.n_s = n_s
        self.init_bingham()
    
    def init_bingham(self):
        self.z_z = nn.Parameter(torch.randn(self.K, 3).float(), requires_grad=True)
        
        self.z_q = nn.Parameter(torch.randn(self.K, 4), requires_grad=True)
        
        self.z_fq = nn.Parameter(torch.randn(self.K, 4), requires_grad=True)
        
    
    @staticmethod
    def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        """Convert a batch of unit quaternions *(w,x,y,z)* into 3×3 matrices."""
        w, x, y, z = q.unbind(-1)
        wx, wy, wz = w * x, w * y, w * z
        xx, xy, xz = x * x, x * y, x * z
        yy, yz = y * y, y * z
        zz = z * z

        R = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)
        ], dim=-1).reshape(q.shape[0], 3, 3)
        return R

    # ---------------------------------------------------------------------
    def _build_Zdiag(self) -> torch.Tensor:
        """Return (K,4) tensor with ordered negative κ‑values for Bingham."""
        dZ = F.softplus(self._z_z)                       # positive      (K,3)
        Z0 = dZ[:, 0:1]
        Z1 = Z0 + dZ[:, 1:2]
        Z2 = Z1 + dZ[:, 2:3]
        Zdiag = torch.cat([Z0, Z1, Z2], dim=-1)   # κ₃, κ₂, κ₁, κ₀
        Zdiag = -1.0 * Zdiag.clamp(1e-12, 500.0)        # all ≤ 0
        return Zdiag                                    # (K,4)

    def _normalize_quaternion(self) -> torch.Tensor:
        """Map z_q → unit quaternion with non‑negative scalar part (w ≥ 0)."""
        q = self._z_q.reshape(-1, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
        q = ((q[:, 0:1] > 0).float() - 0.5) * 2 * q
        return q
    
    def _normalize_free_quaternion(self) -> torch.Tensor:
        """Map z_q → unit quaternion with non‑negative scalar part (w ≥ 0)."""
        q = self._z_fq.reshape(-1, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
        q = ((q[:, 0:1] > 0).float() - 0.5) * 2 * q
        return q
    
    
    @classmethod
    def bingham_rj(cls, E, K):
        lam = -K

        qa = lam.shape[1]
        mu = torch.zeros(qa, device=E.device)
        sigacginv = 1 + 2 * lam
        SigACG = torch.sqrt(1 / (1+2*lam))

        X = torch.zeros_like(K, device=E.device)
        rj = torch.zeros(lam.shape[0], dtype=torch.bool, device=E.device)

        while not rj.all():
            indx = torch.where(rj==0)
            yp = torch.normal(mu, SigACG[indx])
            y = yp / torch.sqrt(torch.sum(yp**2, 1, keepdim=True))
            X[indx] = y

            lratio = -torch.sum(y**2 * lam[indx], 1) - qa/2 * torch.log(torch.tensor([qa], device=E.device)) + 0.5*(qa-1) + qa/2 * torch.log(torch.sum(y**2 * sigacginv[indx], 1))
            rj[indx] = torch.log(torch.rand(len(lratio), device=E.device)) < lratio

        return torch.bmm(E.transpose(1, 2), X.unsqueeze(2)).squeeze()
    
    @classmethod
    def bingham_quaternion_sample(cls, normalized_q_output, Zbatch):

        q0 = normalized_q_output[:, 0].unsqueeze(1)  # [800, 1]
        q1 = normalized_q_output[:, 1].unsqueeze(1)  # [800, 1]
        q2 = normalized_q_output[:, 2].unsqueeze(1)  # [800, 1]
        q3 = normalized_q_output[:, 3].unsqueeze(1)  # [800, 1]
            
        row1 = torch.cat([q0, -q1, -q2, q3], dim=1)
        row2 = torch.cat([q1, q0, q3, q2], dim=1)
        row3 = torch.cat([q2, -q3, q0, -q1], dim=1)
        row4 = torch.cat([q3, q2, -q1, -q0], dim=1)
            
        z_Q = torch.stack([row1, row2, row3, row4], dim=1)
        zero_tensor = torch.ones(Zbatch.size(0), 1, dtype=Zbatch.dtype, device=Zbatch.device)*(-0.000001)
        Z_expanded = torch.cat((zero_tensor, Zbatch), dim=1)
            
        assert torch.all(Z_expanded < 0)
        z = cls.bingham_rj(z_Q, Z_expanded)
            
        return z
    
    def _sample_bingham(self, Z, q):

        Z = Z.repeat(self.n_s, 1)
        q = q.repeat(self.n_s, 1)

        quats = self.bingham_quaternion_sample(q, Z).view(-1, 4)
        
        q = quats.reshape(-1, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
        q = ((q[:, 0:1] > 0).float() - 0.5) * 2 * q
        
        M = q.T @ q
        
        eigval, eigvec = torch.linalg.eigh(M)
        avg_quat = eigvec[:, -1]
        avg_quat = avg_quat / avg_quat.norm()
        return avg_quat.unsqueeze(0)
    
    def update_mode(self, q):
        
        self.z_q.data.copy_(nn.Parameter(q, requires_grad=True))
        
    def forward(self):
        
        self._z_z = self.z_z
        self._z_q = self.z_q.to(self._z_z.device)

        Z = self._build_Zdiag()
        q = self._normalize_quaternion()

        quats = self._sample_bingham(Z, q)  # (K,4)
        B_paras = torch.cat([q, Z, quats], dim=-1)   
        rotated_spheres = self._quat_to_rotmat(quats)  # (K,3,3)
        rotated_spheres_mode = self._quat_to_rotmat(q)
        return rotated_spheres.squeeze(), rotated_spheres_mode.squeeze(), B_paras