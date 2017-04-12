import mxnet as mx

def getPairNet(net=sym, fc_name=fc_name,eta = eta)
	all_layers = sym.get_internals()
    fc = all_layers[layer_name+'_output']
    parLabel= mx.sym.Variable('pairLabel')
    theta = mx.sym.dot(data=fc,mx.sym.tranpose(net),name='theta')
    DLoss= mx.sym.log1p(mx.sym.exp(theta)) -	parLabel *theta
    Dloss=mx.sym.MakeLoss(data=DLoss,name='DLoss')
    s =mx.sym.sign(data=fc)
    QError=eta * mx.sum(mx.sym.square(s-fc))
    QLoss = mx.sym.MakeLoss(data=QError,name='QLoss')
    loss = mx.sym.Concate(data=[Dloss,QLoss],name='loss')
    return loss