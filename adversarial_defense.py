import torch

class FGSM:
    def __init__(self, model, epsilon=0.1):
        """
        Fast Gradient Sign Method for generating adversarial examples
        Args:
            model: The model to attack
            epsilon: Attack strength parameter
        """
        self.model = model
        self.epsilon = epsilon
        
    def generate(self, x, target, loss_fn=None):
        """
        Generate adversarial examples using FGSM
        
        Args:
            x: Input tensor (token indices)
            target: True labels
            loss_fn: Loss function (default: BCELoss after sigmoid)
            
        Returns:
            Perturbed embeddings
        """
        training_mode = self.model.training
        self.model.eval()
        
        emb = self.model.lookup(x)
        avg_norm = torch.mean(torch.norm(emb, dim=-1))
        adaptive_epsilon = self.epsilon * (avg_norm / 10.0) 

        emb_adv = emb.clone().detach().requires_grad_(True)
        
        h_non_static = emb_adv.unsqueeze(1) 
        
        h_list = []
        for i in range(len(self.model.filter_sizes)):
            h_n = self.model.conv_layers[i](h_non_static)
            h_n = h_n.view(h_n.shape[0], 1, h_n.shape[1] * h_n.shape[2])
            h_n = self.model.pool_layers[i](h_n)
            h_n = torch.relu(h_n)
            h_n = h_n.view(h_n.shape[0], -1)
            h_list.append(h_n)
            
        if len(self.model.filter_sizes) > 1:
            h = torch.cat(h_list, 1)
        else:
            h = h_list[0]
            
        h = torch.relu(self.model.l1(h))
        outputs = self.model.l2(h)
        outputs = torch.sigmoid(outputs)
        
        if loss_fn is None:
            loss_fn = torch.nn.BCELoss()
        
        loss = loss_fn(outputs, target)
        
        loss.backward()
        
        perturbed_embeddings = emb + adaptive_epsilon * emb_adv.grad.sign()
        
        if training_mode:
            self.model.train()
        
        return perturbed_embeddings

class FeatureSqueezing:
    def __init__(self, bit_depth=8):
        """
        Feature squeezing defense
        Args:
            bit_depth: Number of bits to use for quantization
        """
        self.bit_depth = bit_depth
        self.max_val = 2 ** bit_depth - 1
        
    def squeeze(self, x):
        """
        Apply feature squeezing to tensor
        Args:
            x: Input tensor
        Returns:
            Squeezed tensor
        """
        x_scaled = (x * self.max_val).round() / self.max_val
        return x_scaled