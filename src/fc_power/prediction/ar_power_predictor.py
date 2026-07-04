"""Power-only autoregression retained strictly as a weak baseline."""
import numpy as np
class RLSAR:
    def __init__(self,order=20,forgetting=.995):self.order=order;self.lam=forgetting;self.beta=np.zeros(order+1);self.P=np.eye(order+1)*100
    def update(self,history,target):
        x=np.r_[1.,np.asarray(history)[-self.order:][::-1]]; g=self.P@x/(self.lam+x@self.P@x);self.beta+=g*(target-x@self.beta);self.P=(self.P-np.outer(g,x)@self.P)/self.lam
    def predict_one(self,history):return float(np.r_[1.,np.asarray(history)[-self.order:][::-1]]@self.beta)

