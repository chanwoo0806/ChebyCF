from torch import nn
from src.module import ChebyFilter, IdealFilter, DegreeNorm, LinearFilter

def build_model(args):
    if args.model.lower() == 'chebycf':
        return ChebyCF(args.K, args.phi, args.eta, args.alpha, args.beta).to(args.device)
    elif args.model.lower() == 'gfcf':
        return GFCF(args.alpha).to(args.device)
    raise NotImplementedError(f'Model named {args.model} is not implemented.')

class AllRankRec(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, observed_inter):
        """
        Predict preference scores for all items given the observed interaction.
        Args:
            observed_inter (torch.Tensor): A binary matrix of shape (batch_size, num_items),
                where 1 indicates an observed interaction.
        Returns:
            pred_score (torch.Tensor): A score matrix of shape (batch_size, num_items),
            where higher values indicate higher predicted preference.
        """
        pass
    
    def mask_observed(self, pred_score, observed_inter):
        # Mask out the scores for items that have been already interacted with.
        return pred_score * (1 - observed_inter) - 1e8 * observed_inter

    def full_predict(self, observed_inter):
        pred_score = self.forward(observed_inter)
        return self.mask_observed(pred_score, observed_inter)

class ChebyCF(AllRankRec):
    def __init__(self, K, phi, eta, alpha, beta):
        super().__init__()        
        self.cheby = ChebyFilter(K, phi)
        self.ideal = IdealFilter(eta, alpha) if eta > 0 and alpha > 0 else None
        self.norm = DegreeNorm(beta) if beta > 0 else None
        
    def fit(self, inter):
        self.cheby.fit(inter)
        if self.ideal:
            self.ideal.fit(inter)
        if self.norm:
            self.norm.fit(inter)
            
    def forward(self, signal):
        if self.norm:
            signal = self.norm.forward_pre(signal)
        output = self.cheby.forward(signal)
        if self.ideal:
            output += self.ideal.forward(signal)
        if self.norm:
            output = self.norm.forward_post(output)
        return output

class GFCF(AllRankRec):
    def __init__(self, alpha):
        super().__init__()
        self.linear = LinearFilter()
        self.ideal = IdealFilter(256, alpha) if alpha > 0 else None
        self.norm = DegreeNorm(0.5)
        
    def fit(self, inter):
        self.linear.fit(inter)
        if self.ideal:
            self.ideal.fit(inter)
        self.norm.fit(inter)
        
    def forward(self, signal):
        output = self.linear(signal)
        if self.ideal:
            signal = self.norm.forward_pre(signal)
            signal = self.ideal(signal)
            signal = self.norm.forward_post(signal)
            output += signal
        return output