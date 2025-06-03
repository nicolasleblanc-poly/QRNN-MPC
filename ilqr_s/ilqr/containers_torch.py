import torch
import numpy as np

class Dynamics:
    def __init__(self, f, model=None):
        """
        Args:
            f: Dynamics function (PyTorch compatible)
            model: Optional neural network model
        """
        self.f = f  # f(x, u, model) -> x_next
        self.model = model

    def predict(self, x_np, u_np):
        """NumPy interface for prediction"""
        x = torch.from_numpy(x_np).float().requires_grad_(True)
        u = torch.from_numpy(u_np).float().requires_grad_(True)
        with torch.no_grad():
            if self.model:
                x_next = self.f(x, u, self.model)
            else:
                x_next = self.f(x, u)
        return x_next.numpy()

    def jacobians(self, x_np, u_np):
        """Compute ∂f/∂x and ∂f/∂u using autograd"""
        x = torch.from_numpy(x_np).float().requires_grad_(True)
        u = torch.from_numpy(u_np).float().requires_grad_(True)
        
        if self.model:
            x_next = self.f(x, u, self.model)
        else:
            x_next = self.f(x, u)

        # Compute Jacobians
        Jx = torch.zeros((x_next.shape[0], x.shape[0]))
        Ju = torch.zeros((x_next.shape[0], u.shape[0]))
        
        for i in range(x_next.shape[0]):
            grad_x = torch.autograd.grad(x_next[i], x, retain_graph=True)[0]
            grad_u = torch.autograd.grad(x_next[i], u, retain_graph=True)[0]
            Jx[i] = grad_x
            Ju[i] = grad_u
            
        return Jx.numpy(), Ju.numpy()

class Cost:
    def __init__(self, L, Lf):
        """
        Args:
            L: Running cost (PyTorch function)
            Lf: Terminal cost (PyTorch function)
        """
        self.L = L
        self.Lf = Lf

    def running_cost(self, x_np, u_np):
        x = torch.from_numpy(x_np).float()
        u = torch.from_numpy(u_np).float()
        with torch.no_grad():
            return self.L(x, u).item()

    def gradients(self, x_np, u_np):
        """Compute all required gradients via autograd"""
        x = torch.from_numpy(x_np).float().requires_grad_(True)
        u = torch.from_numpy(u_np).float().requires_grad_(True)
        
        # First-order gradients
        cost = self.L(x, u)
        L_x = torch.autograd.grad(cost, x, create_graph=True)[0]
        L_u = torch.autograd.grad(cost, u, create_graph=True)[0]
        
        # Second-order gradients
        L_xx = torch.zeros((x.shape[0], x.shape[0]))
        L_ux = torch.zeros((u.shape[0], x.shape[0]))
        L_uu = torch.zeros((u.shape[0], u.shape[0]))
        
        for i in range(x.shape[0]):
            L_xx[i] = torch.autograd.grad(L_x[i], x, retain_graph=True)[0]
        for i in range(u.shape[0]):
            L_ux[i] = torch.autograd.grad(L_u[i], x, retain_graph=True)[0]
            L_uu[i] = torch.autograd.grad(L_u[i], u, retain_graph=True)[0]
        
        return (
            L_x.detach().numpy(),
            L_u.detach().numpy(),
            L_xx.numpy(),
            L_ux.numpy(),
            L_uu.numpy()
        )
    